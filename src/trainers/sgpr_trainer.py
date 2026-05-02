import logging

import torch
from functions.LBFGS import FullBatchLBFGS
from models.sgpr import SGPR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from trainers.acquisition_fn_trainers import EITrainer, LogEITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import (GuacamolTrainer, HartmannTrainer,
                                    LassoDNATrainer, LunarTrainer,
                                    RoverTrainer)
from trainers.utils.turbo import update_state
from trainers.utils.turbo_trainer_mixin import TurboTrainerMixin

import gpytorch
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import InducingPointKernel
from botorch.utils import standardize


class SGPRTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = 'vanilla_sgpr'

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        if self.turn_on_sobol_init:
            train_x, train_y = self.sobol_initialize_data()
        else:
            train_x, train_y = self.initialize_data()

        self.train_y_mean = train_y.mean()
        self.train_y_std = train_y.std()
        if self.train_y_std == 0:
            self.train_y_std = 1

        # log initial y_max
        print(f'initial y max: {train_y.max().item()}')
        logging.info(f'initial y max: {train_y.max().item()}')
        if not self.turn_off_wandb:
            self.tracker.log({
                'initial y max': train_y.max().item(),
                'best reward': train_y.max().item()
            })

        # get inducing points
        # Clone so learned inducing-point updates do not mutate the ExactGP
        # training inputs through shared storage.
        inducing_points = train_x[:self.num_inducing_points].clone()

        if self.norm_data:
            # get normalized train y
            model_train_y = (train_y - self.train_y_mean) / self.train_y_std
        else:
            model_train_y = train_y

        # init model
        self.model = SGPR(
            train_x=train_x,
            train_y=standardize(model_train_y.squeeze())
            if self.turn_on_outcome_transform else model_train_y.squeeze(),
            inducing_points=inducing_points,
            likelihood=gpytorch.likelihoods.GaussianLikelihood().to(
                self.device),
            kernel_type=self.kernel_type,
            kernel_likelihood_prior=self.kernel_likelihood_prior,
            use_ard_kernel=self.use_ard_kernel,
            add_likelihood=self.add_likelihood_to_posterior,
            turn_off_prior=self.turn_off_prior,
            ln_noise_prior_loc=self.ln_noise_prior_loc,
            spherical_linear_lengthscale_prior=self.spherical_linear_lengthscale_prior).to(
                self.device, self.data_type)

        # set custom LR on IP and variational parameters
        variational_params_and_ip = [
            p for name, p in self.model.named_parameters()
            if 'variational' in name
        ]
        others = [
            p for name, p in self.model.named_parameters()
            if 'variational' not in name
        ]

        if self.optimizer_type == torch.optim.Adam:
            self.optimizer = self.optimizer_type(
                [{
                    'params': others
                }, {
                    'params': variational_params_and_ip,
                    'lr': self.svgp_inducing_point_learning_rate
                }],
                lr=self.learning_rate)
        elif self.optimizer_type == torch.optim.LBFGS:
            self.optimizer = self.optimizer_type(self.model.parameters(),
                                                 lr=self.learning_rate,
                                                 line_search_fn='strong_wolfe')
        elif self.optimizer_type == FullBatchLBFGS:
            self.optimizer = self.optimizer_type(self.model.parameters(),
                                                 lr=self.learning_rate,
                                                 dtype=self.data_type)
        else:
            # botorch lbfgs case
            pass

        reward = []
        for i in trange(self.max_oracle_calls - self.num_initial_points):
            if self.norm_data:
                # get normalized train y
                model_train_y = (train_y -
                                 self.train_y_mean) / self.train_y_std
            else:
                model_train_y = train_y

            # only update on recently acquired points
            if i > 0:
                update_x = train_x[-self.update_train_size:]
                # y needs to only have 1 dimension when training in gpytorch
                update_y = model_train_y.squeeze()[-self.update_train_size:]

                if self.turn_on_outcome_transform:
                    update_y = standardize(update_y)

                if self.reinit_hyperparams:
                    self.model.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    ).to(self.device)
                    base_kernel = gpytorch.kernels.MaternKernel(
                        2.5, ard_num_dims=None)

                    covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
                    self.model.covar_module = InducingPointKernel(
                        covar_module,
                        inducing_points=inducing_points.clone(),
                        likelihood=self.model.likelihood)
                if self.reinit_mean:
                    self.model.mean_module = gpytorch.means.ConstantMean()
            else:
                update_x = train_x
                update_y = model_train_y.squeeze()
                if self.turn_on_outcome_transform:
                    update_y = standardize(update_y)

            # Update the ExactGP training data through the model API so cached
            # prediction state stays consistent across BO iterations.
            self.model.set_train_data(inputs=update_x,
                                      targets=update_y,
                                      strict=False)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model)

            if self.optimizer_type == 'botorch_lbfgs':
                self.model.train()
                mll = fit_gpytorch_mll(mll)
                epochs_trained = -1
                final_loss = -1
            else:
                train_loader = self.generate_dataloaders(train_x=update_x,
                                                         train_y=update_y)

                final_loss, epochs_trained = self.train_model(
                    train_loader, mll)
            self.model.eval()

            x_next, x_af_val, origin = self.data_acquisition_iteration(
                self.model, update_y, train_x)

            cos_sim_incum = self.compute_cos_sim_to_incumbent(train_x=train_x,
                                                              train_y=train_y,
                                                              x_next=x_next)
            # Evaluate candidates
            if self.turn_on_simple_input_transform:
                y_next = self.task(x_next * (self.task.ub - self.task.lb) +
                                   self.task.lb)
            else:
                y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            self.log_wandb_metrics(train_y=train_y,
                                   y_next=y_next.item(),
                                   final_loss=final_loss,
                                   train_rmse=-1,
                                   train_nll=-1,
                                   cos_sim_incum=cos_sim_incum,
                                   epochs_trained=epochs_trained,
                                   x_af_val=x_af_val.item(),
                                   x_next_sigma=0,
                                   standardized_gain=0,
                                   candidate_origin=origin)

            reward.append(train_y.max().item())

    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset,
                                  batch_size=train_x.shape[0],
                                  shuffle=False)
        return train_loader


class TurboSGPRTrainer(TurboTrainerMixin, SGPRTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'sgpr_turbo'

    def _select_local_model_data(self, local_train_x, local_train_y,
                                 local_iteration, use_sliding_window):
        if self.norm_data:
            model_train_y = (local_train_y -
                             self.train_y_mean) / self.train_y_std
        else:
            model_train_y = local_train_y

        if use_sliding_window and local_iteration > 0:
            update_x = local_train_x[-self.update_train_size:]
            update_y = model_train_y.squeeze()[-self.update_train_size:]
        else:
            update_x = local_train_x
            update_y = model_train_y.squeeze()

        if self.turn_on_outcome_transform:
            update_y = standardize(update_y)

        return update_x, update_y

    def _init_turbo_model(self, train_x, train_y):
        inducing_points = train_x[:min(self.num_inducing_points,
                                       train_x.size(0))].clone()
        self.model = SGPR(
            train_x=train_x,
            train_y=train_y,
            inducing_points=inducing_points,
            likelihood=gpytorch.likelihoods.GaussianLikelihood().to(
                self.device),
            kernel_type=self.kernel_type,
            kernel_likelihood_prior=self.kernel_likelihood_prior,
            use_ard_kernel=self.use_ard_kernel,
            add_likelihood=self.add_likelihood_to_posterior,
            turn_off_prior=self.turn_off_prior,
            ln_noise_prior_loc=self.ln_noise_prior_loc,
            spherical_linear_lengthscale_prior=self.
            spherical_linear_lengthscale_prior).to(self.device, self.data_type)

        variational_params_and_ip = [
            p for name, p in self.model.named_parameters()
            if 'variational' in name or 'inducing_points' in name
        ]
        others = [
            p for name, p in self.model.named_parameters()
            if 'variational' not in name and 'inducing_points' not in name
        ]

        if self.optimizer_type == torch.optim.Adam:
            self.optimizer = self.optimizer_type(
                [{
                    'params': others
                }, {
                    'params': variational_params_and_ip,
                    'lr': self.svgp_inducing_point_learning_rate
                }],
                lr=self.learning_rate)
        elif self.optimizer_type == torch.optim.LBFGS:
            self.optimizer = self.optimizer_type(self.model.parameters(),
                                                 lr=self.learning_rate,
                                                 line_search_fn='strong_wolfe')
        elif self.optimizer_type == FullBatchLBFGS:
            self.optimizer = self.optimizer_type(self.model.parameters(),
                                                 lr=self.learning_rate,
                                                 dtype=self.data_type)
        else:
            self.optimizer = None

    def run_turbo_experiment(self, iteration: int, use_sliding_window: bool):
        logging.info(self.__dict__)
        if self.turn_on_sobol_init:
            local_train_x, local_train_y = self.sobol_initialize_data()
        else:
            local_train_x, local_train_y = self.initialize_data()
        global_train_x = local_train_x.clone()
        global_train_y = local_train_y.clone()

        self.set_local_train_y_stats(local_train_y)

        print(f'initial y max: {global_train_y.max().item()}')
        logging.info(f'initial y max: {global_train_y.max().item()}')
        if not self.turn_off_wandb:
            self.tracker.log({
                'initial y max': global_train_y.max().item(),
                'best reward': global_train_y.max().item()
            })

        self.initialize_turbo_state(train_x=local_train_x, train_y=local_train_y)
        reward = []
        local_iteration = 0
        initial_num_calls = self.task.num_calls
        total_budget = max(self.max_oracle_calls - initial_num_calls, 0)

        for _ in trange(total_budget):
            if self.task.num_calls >= self.max_oracle_calls:
                break

            update_x, update_y = self._select_local_model_data(
                local_train_x=local_train_x,
                local_train_y=local_train_y,
                local_iteration=local_iteration,
                use_sliding_window=use_sliding_window)
            self._init_turbo_model(update_x, update_y)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model)

            if self.optimizer_type == 'botorch_lbfgs':
                self.model.train()
                fit_gpytorch_mll(mll)
                epochs_trained = -1
                final_loss = -1
            else:
                train_loader = self.generate_dataloaders(train_x=update_x,
                                                         train_y=update_y)
                final_loss, epochs_trained = self.train_model(
                    train_loader, mll)
            self.model.eval()

            x_next, x_af_val, origin = self.data_acquisition_iteration(
                self.model, update_y, update_x)

            cos_sim_incum = self.compute_cos_sim_to_incumbent(train_x=local_train_x,
                                                              train_y=local_train_y,
                                                              x_next=x_next)

            if self.turn_on_simple_input_transform:
                y_next = self.task(x_next * (self.task.ub - self.task.lb) +
                                   self.task.lb)
            else:
                y_next = self.task(x_next)

            local_train_x = torch.cat((local_train_x, x_next), dim=-2)
            local_train_y = torch.cat((local_train_y, y_next), dim=-2)
            global_train_x = torch.cat((global_train_x, x_next), dim=-2)
            global_train_y = torch.cat((global_train_y, y_next), dim=-2)

            self.tr_state = update_state(state=self.tr_state, Y_next=y_next)
            local_iteration += 1

            if self.tr_state.restart_triggered:
                if self.use_faithful_turbo_restart:
                    remaining_budget = self.max_oracle_calls - self.task.num_calls
                    if remaining_budget < self.num_initial_points:
                        self.log_wandb_metrics(train_y=global_train_y,
                                               y_next=y_next.item(),
                                               final_loss=final_loss,
                                               train_rmse=-1,
                                               train_nll=-1,
                                               cos_sim_incum=cos_sim_incum,
                                               epochs_trained=epochs_trained,
                                               x_af_val=x_af_val.item(),
                                               x_next_sigma=0,
                                               standardized_gain=0,
                                               candidate_origin=origin)
                        reward.append(global_train_y.max().item())
                        break

                    local_train_x, local_train_y = self.restart_local_trust_region()
                    global_train_x = torch.cat((global_train_x, local_train_x),
                                               dim=-2)
                    global_train_y = torch.cat((global_train_y, local_train_y),
                                               dim=-2)
                    local_iteration = 0
                else:
                    self.initialize_turbo_state(train_x=local_train_x,
                                                train_y=local_train_y)

            self.log_wandb_metrics(train_y=global_train_y,
                                   y_next=y_next.item(),
                                   final_loss=final_loss,
                                   train_rmse=-1,
                                   train_nll=-1,
                                   cos_sim_incum=cos_sim_incum,
                                   epochs_trained=epochs_trained,
                                   x_af_val=x_af_val.item(),
                                   x_next_sigma=0,
                                   standardized_gain=0,
                                   candidate_origin=origin)

            reward.append(global_train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    def run_experiment(self, iteration: int):
        self.run_turbo_experiment(iteration=iteration, use_sliding_window=False)


class TurboSGPRSlidingWindowTrainer(TurboSGPRTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'sgpr_turbo_sliding_window'

    def run_experiment(self, iteration: int):
        self.run_turbo_experiment(iteration=iteration, use_sliding_window=True)


class OsmbLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class HartmannEISGPRTrainer(SGPRTrainer, HartmannTrainer, EITrainer):
    pass


class LunarEISGPRTrainer(SGPRTrainer, LunarTrainer, EITrainer):
    pass


class RoverEISGPRTrainer(SGPRTrainer, RoverTrainer, EITrainer):
    pass


class RoverLogEISGPRTrainer(SGPRTrainer, RoverTrainer, LogEITrainer):
    pass


class LassoDNALogEISGPRTrainer(SGPRTrainer, LassoDNATrainer, LogEITrainer):
    pass


class FexoLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class Med1LogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med1', **kwargs)


class Med2LogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med2', **kwargs)


class PdopLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='pdop', **kwargs)


class AdipLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='adip', **kwargs)


class RanoLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='rano', **kwargs)


class SigaLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='siga', **kwargs)


class ZaleLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='zale', **kwargs)


class ValtLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='valt', **kwargs)


class DhopLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='dhop', **kwargs)


class ShopLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='shop', **kwargs)


class HartmannEISGPRTurboTrainer(TurboSGPRTrainer, HartmannTrainer, EITrainer):
    pass


class LunarEISGPRTurboTrainer(TurboSGPRTrainer, LunarTrainer, EITrainer):
    pass


class RoverEISGPRTurboTrainer(TurboSGPRTrainer, RoverTrainer, EITrainer):
    pass


class RoverLogEISGPRTurboTrainer(TurboSGPRTrainer, RoverTrainer,
                                 LogEITrainer):
    pass


class LassoDNALogEISGPRTurboTrainer(TurboSGPRTrainer, LassoDNATrainer,
                                    LogEITrainer):
    pass


class OsmbLogEISGPRTurboTrainer(TurboSGPRTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class FexoLogEISGPRTurboTrainer(TurboSGPRTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class Med1LogEISGPRTurboTrainer(TurboSGPRTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med1', **kwargs)


class Med2LogEISGPRTurboTrainer(TurboSGPRTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med2', **kwargs)


class PdopLogEISGPRTurboTrainer(TurboSGPRTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='pdop', **kwargs)


class AdipLogEISGPRTurboTrainer(TurboSGPRTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='adip', **kwargs)


class RanoLogEISGPRTurboTrainer(TurboSGPRTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='rano', **kwargs)


class HartmannEISGPRTurboSlidingWindowTrainer(TurboSGPRSlidingWindowTrainer,
                                              HartmannTrainer, EITrainer):
    pass


class LunarEISGPRTurboSlidingWindowTrainer(TurboSGPRSlidingWindowTrainer,
                                           LunarTrainer, EITrainer):
    pass


class RoverEISGPRTurboSlidingWindowTrainer(TurboSGPRSlidingWindowTrainer,
                                           RoverTrainer, EITrainer):
    pass


class RoverLogEISGPRTurboSlidingWindowTrainer(
        TurboSGPRSlidingWindowTrainer, RoverTrainer, LogEITrainer):
    pass


class LassoDNALogEISGPRTurboSlidingWindowTrainer(
        TurboSGPRSlidingWindowTrainer, LassoDNATrainer, LogEITrainer):
    pass


class OsmbLogEISGPRTurboSlidingWindowTrainer(
        TurboSGPRSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class FexoLogEISGPRTurboSlidingWindowTrainer(
        TurboSGPRSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class Med1LogEISGPRTurboSlidingWindowTrainer(
        TurboSGPRSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med1', **kwargs)


class Med2LogEISGPRTurboSlidingWindowTrainer(
        TurboSGPRSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med2', **kwargs)


class PdopLogEISGPRTurboSlidingWindowTrainer(
        TurboSGPRSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='pdop', **kwargs)


class AdipLogEISGPRTurboSlidingWindowTrainer(
        TurboSGPRSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='adip', **kwargs)


class RanoLogEISGPRTurboSlidingWindowTrainer(
        TurboSGPRSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='rano', **kwargs)
