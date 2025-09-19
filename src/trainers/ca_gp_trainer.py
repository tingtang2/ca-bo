import copy
import logging
import math

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ComputationAwareELBO, ExactMarginalLogLikelihood
from linear_operator import operators
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from functions.LBFGS import FullBatchLBFGS
from models.ca_gp import CaGP
from trainers.acquisition_fn_trainers import EITrainer, LogEITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import (GuacamolTrainer, HartmannTrainer,
                                    LassoDNATrainer, LunarTrainer,
                                    RoverTrainer)
from trainers.svgp_trainer import SVGPEULBOTrainer


class CaGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.train_batch_size = 32

        self.name = 'vanilla_ca_gp'

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()

        # log initial y_max
        print(f'initial y max: {train_y.max().item()}')
        logging.info(f'initial y max: {train_y.max().item()}')
        if not self.turn_off_wandb:
            self.tracker.log({'initial y max': train_y.max().item()})

        self.train_y_mean = train_y.mean()
        self.train_y_std = train_y.std()
        if self.train_y_std == 0:
            self.train_y_std = 1

        reward = []
        for i in trange(self.max_oracle_calls - self.num_initial_points):
            if self.norm_data:
                # get normalized train y
                model_train_y = (train_y -
                                 self.train_y_mean) / self.train_y_std
            else:
                model_train_y = train_y

            if self.static_proj_dim != -1:
                proj_dim = self.static_proj_dim
            else:
                proj_dim = int(self.proj_dim_ratio * train_x.size(0))

            self.model = CaGP(
                train_inputs=train_x,
                train_targets=model_train_y.squeeze(),
                projection_dim=proj_dim,
                likelihood=GaussianLikelihood().to(self.device),
                kernel_type=self.kernel_type,
                init_mode=self.ca_gp_init_mode,
                kernel_likelihood_prior=self.kernel_likelihood_prior,
                use_ard_kernel=self.use_ard_kernel).to(self.device)
            # if self.debug:
            #     torch.save(train_x, f'{self.save_dir}models/train_x.pt')
            #     torch.save(model_train_y,
            #                f'{self.save_dir}models/model_train_y.pt')
            #     torch.save(train_y, f'{self.save_dir}models/train_y.pt')

            action_params = [
                p for name, p in self.model.named_parameters()
                if 'action' in name
            ]
            others = [
                p for name, p in self.model.named_parameters()
                if 'action' not in name
            ]

            self.optimizer = self.optimizer_type(
                [{
                    'params': others
                }, {
                    'params': action_params,
                    'lr': self.ca_gp_actions_learning_rate
                }],
                lr=self.learning_rate)

            mll = ComputationAwareELBO(self.model.likelihood, self.model)
            exact_mll = ExactMarginalLogLikelihood(self.model.likelihood,
                                                   self.model)

            train_loader = self.generate_dataloaders(
                train_x=train_x, train_y=model_train_y.squeeze())

            final_loss, epochs_trained = self.train_model(train_loader, mll)

            # calc gradients of actions
            total_norm = 0.0
            for p in action_params:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
            total_norm = total_norm**0.5

            self.model.eval()

            train_rmse = self.eval(train_x, model_train_y)
            train_nll = self.compute_nll(train_x, model_train_y.squeeze(),
                                         exact_mll)

            x_next = self.data_acquisition_iteration(self.model,
                                                     model_train_y.squeeze(),
                                                     train_x).to(self.device)

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
                                   epochs_trained=epochs_trained,
                                   action_norm=total_norm)

            reward.append(train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset,
                                  batch_size=train_x.shape[0],
                                  shuffle=False)
        return train_loader


class CaGPEULBOTrainer(SVGPEULBOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.early_stopping_threshold = 3
        self.train_batch_size = 32

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()

        # log initial y_max
        print(f'initial y max: {train_y.max().item()}')
        logging.info(f'initial y max: {train_y.max().item()}')
        if not self.turn_off_wandb:
            self.tracker.log({'initial y max': train_y.max().item()})

        self.train_y_mean = train_y.mean()
        self.train_y_std = train_y.std()
        if self.train_y_std == 0:
            self.train_y_std = 1

        reward = []
        for i in trange(self.max_oracle_calls - self.num_initial_points):
            if self.norm_data:
                # get normalized train y
                model_train_y = (train_y -
                                 self.train_y_mean) / self.train_y_std
            else:
                model_train_y = train_y

            if self.static_proj_dim != -1:
                proj_dim = self.static_proj_dim
            else:
                proj_dim = int(self.proj_dim_ratio * train_x.size(0))

            self.model = CaGP(
                train_inputs=train_x,
                train_targets=model_train_y.squeeze(),
                projection_dim=proj_dim,
                likelihood=GaussianLikelihood().to(self.device),
                kernel_type=self.kernel_type,
                init_mode=self.ca_gp_init_mode,
                kernel_likelihood_prior=self.kernel_likelihood_prior,
                use_ard_kernel=self.use_ard_kernel).to(self.device)

            self.optimizer = self.optimizer_type(
                [{
                    'params': self.model.parameters(),
                }], lr=self.learning_rate)

            mll = ComputationAwareELBO(self.model.likelihood, self.model)
            exact_mll = ExactMarginalLogLikelihood(self.model.likelihood,
                                                   self.model)

            train_loader = self.generate_dataloaders(
                train_x=train_x, train_y=model_train_y.squeeze())

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            x_next = self.data_acquisition_iteration(self.model, model_train_y,
                                                     train_x)

            # above is warm start
            torch.autograd.set_detect_anomaly(True)

            n_failures = 0
            success = False
            model_state_before_update = copy.deepcopy(self.model.state_dict())

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

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    def train_model(self, train_loader: DataLoader, mll):
        self.model.train()
        for i in range(self.epochs):
            loss = self.train_epoch(train_loader, mll)
        return loss, i + 1

    def train_epoch(self, train_loader: DataLoader, mll):
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            self.optimizer.zero_grad()

            output = self.model(x.to(self.device))
            loss = -mll(output, y.double().to(self.device))

            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.grad_clip)

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss

    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset,
                                  batch_size=train_x.shape[0],
                                  shuffle=False)
        return train_loader


class CaGPSlidingWindowTrainer(CaGPTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = 'sliding_window_ca_gp'

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()

        # log initial y_max
        print(f'initial y max: {train_y.max().item()}')
        logging.info(f'initial y max: {train_y.max().item()}')
        if not self.turn_off_wandb:
            self.tracker.log({'initial y max': train_y.max().item()})

        self.train_y_mean = train_y.mean()
        self.train_y_std = train_y.std()
        if self.train_y_std == 0:
            self.train_y_std = 1

        if self.static_proj_dim != -1:
            proj_dim = self.static_proj_dim
        else:
            proj_dim = int(self.proj_dim_ratio * train_x.size(0))

        if self.norm_data:
            # get normalized train y
            model_train_y = (train_y - self.train_y_mean) / self.train_y_std
        else:
            model_train_y = train_y

        # add on actions if proj dim > num initial points
        if proj_dim > self.num_initial_points:
            initial_proj_dim = self.num_initial_points
        else:
            initial_proj_dim = proj_dim

        self.model = CaGP(
            train_inputs=train_x,
            train_targets=model_train_y.squeeze(),
            projection_dim=initial_proj_dim,
            likelihood=GaussianLikelihood().to(self.device),
            kernel_type=self.kernel_type,
            init_mode=self.ca_gp_init_mode,
            kernel_likelihood_prior=self.kernel_likelihood_prior,
            use_ard_kernel=self.use_ard_kernel,
            standardize_outputs=self.turn_on_outcome_transform).to(self.device)
        # if self.debug:
        #     torch.save(train_x, f'{self.save_dir}models/train_x.pt')
        #     torch.save(model_train_y,
        #                f'{self.save_dir}models/model_train_y.pt')
        #     torch.save(train_y, f'{self.save_dir}models/train_y.pt')
        if self.freeze_actions:
            self.model.actions_op.blocks.data = torch.ones(
                (train_x.shape[0], self.model.num_non_zero))
            self.model.actions_op.blocks.requires_grad = False

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

                if self.add_actions_by_reinit and train_x.size(0) <= proj_dim:
                    self.model = CaGP(
                        train_inputs=train_x,
                        train_targets=update_y,
                        projection_dim=min(update_y.size(0), proj_dim),
                        likelihood=GaussianLikelihood().to(self.device),
                        kernel_type=self.kernel_type,
                        init_mode=self.ca_gp_init_mode,
                        kernel_likelihood_prior=self.kernel_likelihood_prior,
                        use_ard_kernel=self.use_ard_kernel,
                        standardize_outputs=self.turn_on_outcome_transform).to(
                            self.device)
                else:
                    # set projection dim to min of training data size and requested dim size
                    self.model.projection_dim = min(update_y.size(0), proj_dim)

                    # sliding window here
                    # Set number of non-zero action entries such that num_non_zero * projection_dim = num_train_targets
                    self.model.num_non_zero = update_y.size(
                        -1) // self.model.projection_dim

                    self.model.train_inputs = tuple(
                        tri.unsqueeze(-1) if tri.ndimension() == 1 else tri
                        for tri in (update_x[0:self.model.num_non_zero *
                                             self.model.projection_dim], ))

                    if self.turn_on_outcome_transform:
                        # need to restandardize the outcomes here
                        self.model.outcome_transform.train()
                        train_targets, train_Yvar = self.model.outcome_transform(
                            Y=update_y[0:self.model.num_non_zero *
                                       self.model.projection_dim].unsqueeze(1),
                            Yvar=None,
                            X=self.model.train_inputs[0])
                        self.model.train_targets = train_targets.squeeze()
                    else:
                        self.model.train_targets = update_y[
                            0:self.model.num_non_zero *
                            self.model.projection_dim]

                    # add on a new action if proj_dim >= training data size, else slide window
                    if train_x.size(0) <= proj_dim:
                        blocks = torch.concat(
                            (self.model.actions_op.blocks.data,
                             torch.randn((1, self.model.num_non_zero)).div(
                                 math.sqrt(self.model.num_non_zero))))
                        non_zero_idcs = torch.arange(
                            self.model.num_non_zero *
                            self.model.projection_dim,
                            device=self.device).reshape(
                                self.model.projection_dim, -1)

                        self.model.non_zero_action_entries = torch.nn.Parameter(
                            blocks)
                        self.model.actions_op = operators.BlockDiagonalSparseLinearOperator(
                            non_zero_idcs=non_zero_idcs,
                            blocks=self.model.non_zero_action_entries,
                            size_input_dim=self.model.num_non_zero *
                            self.model.projection_dim)

                    else:
                        if self.freeze_actions:
                            pass
                        elif self.model.num_non_zero == 1 or not self.roll_actions:
                            if self.non_zero_action_init:
                                new_action = 2 * torch.rand((1, 1)) - 1

                                cast_threshold = 1
                                new_action = torch.where(
                                    (new_action > 0) &
                                    (new_action < cast_threshold),
                                    cast_threshold, new_action)

                                new_action = torch.where(
                                    (new_action < 0) &
                                    (new_action > -cast_threshold),
                                    -cast_threshold, new_action)

                                self.model.actions_op.blocks.data = torch.concat(
                                    (self.model.actions_op.blocks.data[1:],
                                     new_action))
                            else:
                                self.model.actions_op.blocks.data = torch.concat(
                                    (self.model.actions_op.blocks.data[1:],
                                     torch.randn(
                                         (1, self.model.num_non_zero)).div(
                                             math.sqrt(
                                                 self.model.num_non_zero))))

                        else:
                            # roll non zero idcs over 1 to create snaking/platforming effect
                            self.model.actions_op.non_zero_idcs = self.model.actions_op.non_zero_idcs.roll(
                                1)

                            # reinitialize oldest action entry
                            # doing this in a super janky way to not mess with gradients too much
                            new_placeholder = torch.Tensor(
                                self.model.actions_op.blocks.data)
                            new_placeholder[0, 0] = torch.randn(
                                (1)).div(math.sqrt(self.model.num_non_zero))
                            self.model.actions_op.blocks.data.copy_(
                                new_placeholder)

                        if self.reinit_hyperparams:
                            self.model.likelihood = GaussianLikelihood().to(
                                self.device)
                            base_kernel = gpytorch.kernels.MaternKernel(
                                2.5, ard_num_dims=None)

                            covar_module = gpytorch.kernels.ScaleKernel(
                                base_kernel)
                            self.model.covar_module = covar_module
                        if self.reinit_mean:
                            self.model.mean_module = gpytorch.means.ConstantMean(
                            )

            else:
                update_x = train_x
                update_y = model_train_y.squeeze()

            action_params = [
                p for name, p in self.model.named_parameters()
                if 'action' in name
            ]
            others = [
                p for name, p in self.model.named_parameters()
                if 'action' not in name
            ]

            if self.debug:
                # check all params are being optimized
                assert action_params[0].size(0) == self.model.projection_dim
                old_final_action = self.model.actions_op.blocks.data[
                    -1].detach().clone()

            if self.optimizer_type == torch.optim.Adam:
                self.optimizer = self.optimizer_type(
                    [{
                        'params': others
                    }, {
                        'params': action_params,
                        'lr': self.ca_gp_actions_learning_rate
                    }],
                    lr=self.learning_rate)
            elif self.optimizer_type == torch.optim.LBFGS:
                self.optimizer = self.optimizer_type(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    line_search_fn='strong_wolfe')
            elif self.optimizer_type == FullBatchLBFGS:
                self.optimizer = self.optimizer_type(self.model.parameters(),
                                                     lr=self.learning_rate,
                                                     dtype=self.data_type)
            else:
                # botorch lbfgs case
                pass

            exact_mll = ExactMarginalLogLikelihood(self.model.likelihood,
                                                   self.model)

            if self.optimizer_type == 'botorch_lbfgs':
                self.model.train()
                mll = ComputationAwareELBO(self.model.likelihood,
                                           self.model,
                                           return_elbo_terms=False)
                mll = fit_gpytorch_mll(mll)
                epochs_trained = -1
                final_loss = -1
            else:
                mll = ComputationAwareELBO(self.model.likelihood,
                                           self.model,
                                           return_elbo_terms=True)
                if self.turn_on_outcome_transform:
                    train_loader = self.generate_dataloaders(
                        train_x=update_x,
                        train_y=self.model.outcome_transform(
                            update_y.unsqueeze(1))[0].squeeze())
                else:
                    train_loader = self.generate_dataloaders(train_x=update_x,
                                                             train_y=update_y)

                final_loss, epochs_trained = self.train_model(
                    train_loader, mll)

            if self.debug:
                # check final action is optimized
                assert torch.ne(old_final_action,
                                self.model.actions_op.blocks.data[-1])
                print(old_final_action, self.model.actions_op.blocks.data[-1])
            if self.freeze_actions:
                assert (self.model.actions_op.blocks.data == torch.ones(
                    (update_x.size(0), 1))).all()

            # calc gradients of actions
            if not self.freeze_actions:
                total_norm = 0.0
                for p in action_params:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item()**2
                total_norm = total_norm**0.5
            else:
                total_norm = -1
            self.model.eval()

            train_rmse = self.eval(train_x, model_train_y)
            train_nll = self.compute_nll(train_x, model_train_y.squeeze(),
                                         exact_mll)

            x_next, x_af_val = self.data_acquisition_iteration(
                self.model, model_train_y.squeeze(), train_x)

            cos_sim_incum = self.compute_cos_sim_to_incumbent(train_x=train_x,
                                                              train_y=train_y,
                                                              x_next=x_next)

            x_next_mu, x_next_sigma = self.calc_predictive_mean_and_std(
                model=self.model, test_point=x_next)

            standardized_gain = (x_next_mu - torch.max(train_y)) / x_next_sigma

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
                                   epochs_trained=epochs_trained,
                                   action_norm=total_norm,
                                   x_af_val=x_af_val.item(),
                                   x_next_sigma=x_next_sigma.item(),
                                   standardized_gain=standardized_gain.item())

            reward.append(train_y.max().item())

        # self.save_metrics(metrics=reward,
        #                   iter=iteration,
        #                   name=self.trainer_type)


class HartmannEICaGPTrainer(CaGPTrainer, HartmannTrainer, EITrainer):
    pass


class HartmannLogEICaGPTrainer(CaGPTrainer, HartmannTrainer, LogEITrainer):
    pass


class HartmannEICaGPEULBOTrainer(CaGPEULBOTrainer, HartmannTrainer, EITrainer):
    pass


class LunarEICaGPTrainer(CaGPTrainer, LunarTrainer, EITrainer):
    pass


class LunarLogEICaGPTrainer(CaGPTrainer, LunarTrainer, LogEITrainer):
    pass


class LunarEICaGPEULBOTrainer(CaGPEULBOTrainer, LunarTrainer, EITrainer):
    pass


class LunarLogEICaGPEULBOTrainer(CaGPEULBOTrainer, LunarTrainer, LogEITrainer):
    pass


class RoverEICaGPTrainer(CaGPTrainer, RoverTrainer, EITrainer):
    pass


class RoverEICaGPEULBOTrainer(CaGPEULBOTrainer, RoverTrainer, EITrainer):
    pass


class RoverEICaGPSlidingWindowTrainer(CaGPSlidingWindowTrainer, RoverTrainer,
                                      EITrainer):
    pass


class LassoDNALogEICaGPTrainer(CaGPTrainer, LassoDNATrainer, LogEITrainer):
    pass


class LassoDNALogEICaGPSlidingWindowTrainer(CaGPSlidingWindowTrainer,
                                            LassoDNATrainer, LogEITrainer):
    pass


class OsmbLogEICaGPTrainer(CaGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class OsmbLogEICaGPSlidingWindowTrainer(CaGPSlidingWindowTrainer,
                                        GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class FexoLogEICaGPSlidingWindowTrainer(CaGPSlidingWindowTrainer,
                                        GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class Med1LogEICaGPSlidingWindowTrainer(CaGPSlidingWindowTrainer,
                                        GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med1', **kwargs)


class Med2LogEICaGPSlidingWindowTrainer(CaGPSlidingWindowTrainer,
                                        GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med2', **kwargs)
