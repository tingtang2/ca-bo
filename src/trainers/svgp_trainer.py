import copy
import logging

import torch
from models.svgp import SVGPModel
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from trainers.acquisition_fn_trainers import EITrainer, LogEITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import (GuacamolTrainer, HartmannTrainer,
                                    LassoDNATrainer, LunarTrainer,
                                    RoverTrainer)
from trainers.utils.turbo import update_state
from trainers.utils.turbo_trainer_mixin import TurboTrainerMixin
from trainers.utils.expected_log_utility import get_expected_log_utility_ei
from trainers.utils.moss_et_al_inducing_pts_init import \
    GreedyImprovementReduction

import gpytorch
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from botorch.utils import standardize


class SVGPRetrainTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.early_stopping_threshold = 3
        self.train_batch_size = 32

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()

        reward = []
        for i in trange(self.max_oracle_calls - self.num_initial_points):
            if self.norm_data:
                # get normalized train y
                train_y_mean = train_y.mean()
                train_y_std = train_y.std()
                if train_y_std == 0:
                    train_y_std = 1
                train_y = (train_y - train_y_mean) / train_y_std
            # reinit model
            self.model = SVGPModel(
                inducing_points=train_x[-(train_x.size(0) // 2):],
                likelihood=GaussianLikelihood().to(self.device),
                kernel_type=self.kernel_type).to(self.device)

            self.optimizer = self.optimizer_type(
                [{
                    'params': self.model.parameters(),
                }], lr=self.learning_rate)

            mll = VariationalELBO(self.model.likelihood,
                                  self.model,
                                  num_data=train_x.size(0))

            train_loader = self.generate_dataloaders(train_x=train_x,
                                                     train_y=train_y.squeeze())

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            x_next = self.data_acquisition_iteration(self.model, train_y,
                                                     train_x)

            # Evaluate candidates
            y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            logging.info(
                f'Num oracle calls: {self.task.num_calls - 1}, best reward: {train_y.max().item():.3f}, final svgp loss: {final_loss:.3f}, epochs trained: {epochs_trained}, noise param: {self.model.likelihood.noise.item():.6f}'
            )

            if not self.turn_off_wandb:
                self.log_wandb_metrics(train_y=train_y,
                                       final_loss=final_loss,
                                       epochs_trained=epochs_trained)
            reward.append(train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    def train_model(self, train_loader: DataLoader, mll):
        self.model.train()
        best_loss = 1e+5
        early_stopping_counter = 0
        for i in range(self.epochs):
            loss = self.train_epoch(train_loader, mll)

            if loss < best_loss:
                # self.save_model(f'{self.name}_{iter}')
                early_stopping_counter = 0
                best_loss = loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                return loss, i + 1

        return loss, i + 1

    def train_epoch(self, train_loader: DataLoader, mll):
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            self.optimizer.zero_grad()

            output = self.model(x.to(self.device))
            loss = -mll(output, y.to(self.device))

            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.grad_clip)

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss

    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_loader

    def eval(self):
        pass

    def get_optimal_inducing_points(self, prev_inducing_points):
        greedy_imp_reduction = GreedyImprovementReduction(
            model=self.model,
            maximize=True,
        )
        optimal_inducing_points = greedy_imp_reduction.allocate_inducing_points(
            inputs=prev_inducing_points.to(self.device),
            covar_module=self.model.covar_module,
            num_inducing=prev_inducing_points.shape[0],
            input_batch_shape=1,
        )
        return optimal_inducing_points


class SVGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.train_batch_size = 32
        self.name = 'vanilla_svgp'

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
        inducing_points = train_x[:self.num_inducing_points]

        # init model
        self.model = SVGPModel(
            inducing_points=inducing_points,
            likelihood=GaussianLikelihood().to(self.device),
            kernel_type=self.kernel_type,
            kernel_likelihood_prior=self.kernel_likelihood_prior,
            use_ard_kernel=self.use_ard_kernel,
            add_likelihood=self.add_likelihood_to_posterior,
            ln_noise_prior_loc=self.ln_noise_prior_loc,
            spherical_linear_lengthscale_prior=self.spherical_linear_lengthscale_prior,
            turn_off_prior=self.turn_off_prior).to(
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

        self.optimizer = self.optimizer_type(
            [{
                'params': others
            }, {
                'params': variational_params_and_ip,
                'lr': self.svgp_inducing_point_learning_rate
            }],
            lr=self.learning_rate)

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
                if self.reinit_hyperparams:
                    self.model.likelihood = GaussianLikelihood().to(
                        self.device)
                    base_kernel = gpytorch.kernels.MaternKernel(
                        2.5, ard_num_dims=None)

                    covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
                    self.model.covar_module = covar_module
                if self.reinit_mean:
                    self.model.mean_module = gpytorch.means.ConstantMean()
            else:
                update_x = train_x
                update_y = model_train_y.squeeze()

            # need this for RAASP sampling
            self.model.train_inputs = tuple(
                tri.unsqueeze(-1) if tri.ndimension() == 1 else tri
                for tri in (update_x, ))
            if self.turn_on_outcome_transform:
                train_targets = standardize(update_y)
            else:
                train_targets = update_y

            mll = VariationalELBO(self.model.likelihood,
                                  self.model,
                                  num_data=update_x.size(0))

            train_loader = self.generate_dataloaders(
                train_x=update_x, train_y=train_targets.squeeze())

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            x_next, x_af_val, origin = self.data_acquisition_iteration(
                self.model,
                standardize(update_y).squeeze()
                if self.turn_on_outcome_transform else update_y.squeeze(),
                train_x)

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

        # self.save_metrics(metrics=reward,
        #                   iter=iteration,
        #                   name=self.trainer_type)

    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator(device=self.device))
        return train_loader

    def get_optimal_inducing_points(self, prev_inducing_points):
        greedy_imp_reduction = GreedyImprovementReduction(
            model=self.model,
            maximize=True,
        )
        optimal_inducing_points = greedy_imp_reduction.allocate_inducing_points(
            inputs=prev_inducing_points.to(self.device),
            covar_module=self.model.covar_module,
            num_inducing=prev_inducing_points.shape[0],
            input_batch_shape=1,
        )
        return optimal_inducing_points


class HartmannEISVGPTrainer(SVGPTrainer, HartmannTrainer, EITrainer):
    pass


class LunarEISVGPTrainer(SVGPTrainer, LunarTrainer, EITrainer):
    pass


class RoverEISVGPTrainer(SVGPTrainer, RoverTrainer, EITrainer):
    pass


class RoverLogEISVGPTrainer(SVGPTrainer, RoverTrainer, LogEITrainer):
    pass


class HartmannEISVGPRetrainTrainer(SVGPRetrainTrainer, HartmannTrainer,
                                   EITrainer):
    pass


class SVGPEULBOTrainer(SVGPTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.early_stopping_threshold = self.epochs
        self.early_stopping_threshold_eulbo = 3
        self.train_batch_size = 32

        self.inducing_pt_init_w_moss23 = True

        self.alternate_updates = True
        self.x_next_lr = 0.001

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()
        self.train_y_mean = train_y.mean()
        self.train_y_std = train_y.std()
        if self.train_y_std == 0:
            self.train_y_std = 1

        # get inducing points
        inducing_points = train_x[:self.num_inducing_points]

        # init model
        self.model = SVGPModel(inducing_points=inducing_points,
                               likelihood=GaussianLikelihood().to(self.device),
                               kernel_type=self.kernel_type).to(self.device)

        if self.inducing_pt_init_w_moss23:
            optimal_inducing_points = self.get_optimal_inducing_points(
                prev_inducing_points=inducing_points)
            self.model = SVGPModel(
                inducing_points=optimal_inducing_points,
                likelihood=GaussianLikelihood().to(self.device),
                kernel_type=self.kernel_type).to(self.device)

        self.optimizer = self.optimizer_type(
            [{
                'params': self.model.parameters(),
            }], lr=self.learning_rate)

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
            else:
                update_x = train_x
                update_y = model_train_y.squeeze()

            mll = VariationalELBO(self.model.likelihood,
                                  self.model,
                                  num_data=update_x.size(0))

            train_loader = self.generate_dataloaders(train_x=update_x,
                                                     train_y=update_y)

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            x_next = self.data_acquisition_iteration(self.model, model_train_y,
                                                     train_x)

            # above is warm start
            torch.autograd.set_detect_anomaly(True)

            n_failures = 0
            success = False
            model_state_before_update = copy.deepcopy(self.model.state_dict())

            mll = VariationalELBO(self.model.likelihood,
                                  self.model,
                                  num_data=update_x.size(0))
            exact_mll = ExactMarginalLogLikelihood(self.model.likelihood,
                                                   self.model)

            while (n_failures < 8) and (not success):
                try:
                    x_next, final_loss, epochs_trained = self.eulbo_train_model(
                        mll=mll,
                        loader=train_loader,
                        normed_best_train_y=model_train_y.max(),
                        init_x_next=x_next.to(self.device))
                    success = True
                except Exception as e:
                    # decrease lr to stabalize training
                    error_message = e
                    logging.info(f'adjusting lrs: {e}')
                    n_failures += 1
                    self.learning_rate = self.learning_rate / 10
                    self.x_next_lr = self.x_next_lr / 10
                    self.model.load_state_dict(
                        copy.deepcopy(model_state_before_update))
            if not success:
                assert 0, f"\nFailed to complete EULBO model update due to the following error:\n{error_message}"
            self.model.eval()

            train_rmse = self.eval(train_x, model_train_y.squeeze())
            train_nll = self.compute_nll(train_x, model_train_y.squeeze(),
                                         exact_mll)

            cos_sim_incum = self.compute_cos_sim_to_incumbent(train_x=train_x,
                                                              train_y=train_y,
                                                              x_next=x_next)

            # Evaluate candidates
            y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            self.log_wandb_metrics(train_y=train_y,
                                   y_next=y_next.item(),
                                   final_loss=final_loss,
                                   train_rmse=train_rmse,
                                   train_nll=train_nll,
                                   cos_sim_incum=cos_sim_incum,
                                   epochs_trained=epochs_trained)
            reward.append(train_y.max().item())

        # self.save_metrics(metrics=reward,
        #                   iter=iteration,
        #                   name=self.trainer_type)

    def eulbo_train_model(self, loader, mll, normed_best_train_y, init_x_next):
        self.model.train()
        init_x_next = copy.deepcopy(init_x_next)
        x_next = Variable(init_x_next, requires_grad=True)

        model_params_to_update = list(self.model.parameters())
        self.x_next_optimizer = torch.optim.Adam([
            {
                'params': x_next
            },
        ],
                                                 lr=self.x_next_lr)
        self.model_optimizer = torch.optim.Adam(
            [{
                'params': model_params_to_update,
            }], lr=self.learning_rate)
        self.joint_optimizer = torch.optim.Adam(
            [{
                'params': x_next,
            }, {
                'params': model_params_to_update,
            }],
            lr=self.learning_rate)

        best_loss = 1e+5
        early_stopping_counter = 0
        currently_training_model = True
        for i in range(self.eulbo_epochs):
            loss = self.eulbo_train_epoch(
                loader,
                mll,
                x_next,
                currently_training_model=currently_training_model,
                normed_best_train_y=normed_best_train_y)

            currently_training_model = not currently_training_model
            if loss < best_loss:
                # self.save_model(f'{self.name}_{iter}')
                early_stopping_counter = 0
                best_loss = loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold_eulbo:
                return x_next.detach(), loss, i + 1

        return x_next.detach(), loss, i + 1

    def eulbo_train_epoch(self, loader, mll, x_next, currently_training_model,
                          normed_best_train_y):
        total_loss = 0
        for i, (inputs, scores) in enumerate(loader):
            if self.alternate_updates:
                self.model_optimizer.zero_grad()
                self.x_next_optimizer.zero_grad()
            else:
                self.joint_optimizer.zero_grad()
            output = self.model(inputs.to(self.device))
            nelbo = -mll(output, scores.to(self.device))

            expected_log_utility_x_next = get_expected_log_utility_ei(
                self.model,
                best_f=normed_best_train_y,
                x_next=x_next,
                device=self.device)
            loss = nelbo - expected_log_utility_x_next
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.grad_clip)
                torch.nn.utils.clip_grad_norm_(x_next, max_norm=self.grad_clip)
            if self.alternate_updates:
                if currently_training_model:
                    self.model_optimizer.step()
                else:
                    self.x_next_optimizer.step()
            else:
                self.joint_optimizer.step()

            with torch.no_grad():
                x_next[:, :] = x_next.clamp(self.task.lb, self.task.ub)
                total_loss += loss.item()

        return total_loss

    def train_model(self, train_loader: DataLoader, mll):
        self.model.train()
        for i in range(self.epochs):
            loss = self.train_epoch(train_loader, mll)
        return loss, i + 1


class HartmannEISVGPEULBOTrainer(SVGPEULBOTrainer, HartmannTrainer, EITrainer):
    pass


class LunarEISVGPEULBOTrainer(SVGPEULBOTrainer, LunarTrainer, EITrainer):
    pass


class RoverEISVGPEULBOTrainer(SVGPEULBOTrainer, RoverTrainer, EITrainer):
    pass


class LassoDNALogEISVGPTrainer(SVGPTrainer, LassoDNATrainer, LogEITrainer):
    pass


class OsmbLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class FexoLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class Med1LogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med1', **kwargs)


class Med2LogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med2', **kwargs)


class PdopLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='pdop', **kwargs)


class AdipLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='adip', **kwargs)


class RanoLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='rano', **kwargs)


class SigaLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='siga', **kwargs)


class TurboSVGPTrainer(TurboTrainerMixin, SVGPTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'svgp_turbo'

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
            train_targets = standardize(update_y).squeeze()
        else:
            train_targets = update_y.squeeze()

        return update_x, train_targets

    def _init_turbo_model(self, train_x):
        inducing_points = train_x[:min(self.num_inducing_points,
                                       train_x.size(0))].clone()
        self.model = SVGPModel(
            inducing_points=inducing_points,
            likelihood=GaussianLikelihood().to(self.device),
            kernel_type=self.kernel_type,
            kernel_likelihood_prior=self.kernel_likelihood_prior,
            use_ard_kernel=self.use_ard_kernel,
            add_likelihood=self.add_likelihood_to_posterior,
            ln_noise_prior_loc=self.ln_noise_prior_loc,
            spherical_linear_lengthscale_prior=self.
            spherical_linear_lengthscale_prior,
            turn_off_prior=self.turn_off_prior).to(self.device, self.data_type)

        variational_params_and_ip = [
            p for name, p in self.model.named_parameters()
            if 'variational' in name or 'inducing_points' in name
        ]
        others = [
            p for name, p in self.model.named_parameters()
            if 'variational' not in name and 'inducing_points' not in name
        ]

        self.optimizer = self.optimizer_type(
            [{
                'params': others
            }, {
                'params': variational_params_and_ip,
                'lr': self.svgp_inducing_point_learning_rate
            }],
            lr=self.learning_rate)

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

            update_x, train_targets = self._select_local_model_data(
                local_train_x=local_train_x,
                local_train_y=local_train_y,
                local_iteration=local_iteration,
                use_sliding_window=use_sliding_window)
            self._init_turbo_model(update_x)

            self.model.train_inputs = tuple(
                tri.unsqueeze(-1) if tri.ndimension() == 1 else tri
                for tri in (update_x, ))

            mll = VariationalELBO(self.model.likelihood,
                                  self.model,
                                  num_data=update_x.size(0))
            train_loader = self.generate_dataloaders(train_x=update_x,
                                                     train_y=train_targets)
            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            x_next, x_af_val, origin = self.data_acquisition_iteration(
                self.model, train_targets, update_x)

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


class TurboSVGPSlidingWindowTrainer(TurboSVGPTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'svgp_turbo_sliding_window'

    def run_experiment(self, iteration: int):
        self.run_turbo_experiment(iteration=iteration, use_sliding_window=True)


class HartmannEISVGPTurboTrainer(TurboSVGPTrainer, HartmannTrainer,
                                 EITrainer):
    pass


class LunarEISVGPTurboTrainer(TurboSVGPTrainer, LunarTrainer, EITrainer):
    pass


class RoverEISVGPTurboTrainer(TurboSVGPTrainer, RoverTrainer, EITrainer):
    pass


class RoverLogEISVGPTurboTrainer(TurboSVGPTrainer, RoverTrainer,
                                 LogEITrainer):
    pass


class LassoDNALogEISVGPTurboTrainer(TurboSVGPTrainer, LassoDNATrainer,
                                    LogEITrainer):
    pass


class OsmbLogEISVGPTurboTrainer(TurboSVGPTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class FexoLogEISVGPTurboTrainer(TurboSVGPTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class Med1LogEISVGPTurboTrainer(TurboSVGPTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med1', **kwargs)


class Med2LogEISVGPTurboTrainer(TurboSVGPTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med2', **kwargs)


class PdopLogEISVGPTurboTrainer(TurboSVGPTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='pdop', **kwargs)


class AdipLogEISVGPTurboTrainer(TurboSVGPTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='adip', **kwargs)


class RanoLogEISVGPTurboTrainer(TurboSVGPTrainer, GuacamolTrainer,
                                LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='rano', **kwargs)


class HartmannEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, HartmannTrainer, EITrainer):
    pass


class LunarEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, LunarTrainer, EITrainer):
    pass


class RoverEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, RoverTrainer, EITrainer):
    pass


class RoverLogEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, RoverTrainer, LogEITrainer):
    pass


class LassoDNALogEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, LassoDNATrainer, LogEITrainer):
    pass


class OsmbLogEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class FexoLogEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class Med1LogEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med1', **kwargs)


class Med2LogEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med2', **kwargs)


class PdopLogEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='pdop', **kwargs)


class AdipLogEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='adip', **kwargs)


class RanoLogEISVGPTurboSlidingWindowTrainer(
        TurboSVGPSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='rano', **kwargs)


class ZaleLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='zale', **kwargs)


class ValtLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='valt', **kwargs)


class DhopLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='dhop', **kwargs)


class ShopLogEISVGPTrainer(SVGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='shop', **kwargs)
