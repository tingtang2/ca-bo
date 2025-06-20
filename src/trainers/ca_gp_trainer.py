import copy
import logging
import math

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ComputationAwareELBO, ExactMarginalLogLikelihood
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from models.ca_gp import CaGP
from trainers.acquisition_fn_trainers import EITrainer, LogEITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import (HartmannTrainer, LassoDNATrainer,
                                    LunarTrainer, RoverTrainer,
                                    GuacamolTrainer)
from trainers.svgp_trainer import SVGPEULBOTrainer


class CaGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_inducing_points = 100
        self.grad_clip = 2.0

        self.train_batch_size = 32

        self.update_train_size = 100
        self.name = 'vanilla_ca_gp'
        self.debug = False

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
            if self.debug:
                torch.save(train_x, f'{self.save_dir}models/train_x.pt')
                torch.save(model_train_y,
                           f'{self.save_dir}models/model_train_y.pt')
                torch.save(train_y, f'{self.save_dir}models/train_y.pt')

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

    def train_model(self, train_loader: DataLoader, mll):
        self.model.train()
        best_loss = 1e+5
        early_stopping_counter = 0
        best_model_state = None
        for i in range(self.epochs):
            loss = self.train_epoch(train_loader, mll)

            if loss < best_loss:
                # self.save_model(f'{self.name}')
                best_model_state = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
                best_loss = loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                # Load the best model weights before returning
                self.model.load_state_dict(best_model_state)
                return loss, i + 1

        self.model.load_state_dict(best_model_state)
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


class CaGPEULBOTrainer(SVGPEULBOTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_inducing_points = 100
        self.grad_clip = 2.0

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

            self.model = CaGP(train_inputs=train_x,
                              train_targets=model_train_y.squeeze(),
                              projection_dim=proj_dim,
                              likelihood=GaussianLikelihood().to(self.device),
                              kernel_type=self.kernel_type,
                              init_mode=self.ca_gp_init_mode).to(self.device)

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
        self.update_train_size = 100

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

        self.model = CaGP(train_inputs=train_x,
                          train_targets=model_train_y.squeeze(),
                          projection_dim=proj_dim,
                          likelihood=GaussianLikelihood().to(self.device),
                          kernel_type=self.kernel_type,
                          init_mode=self.ca_gp_init_mode).to(self.device)
        if self.debug:
            torch.save(train_x, f'{self.save_dir}models/train_x.pt')
            torch.save(model_train_y,
                       f'{self.save_dir}models/model_train_y.pt')
            torch.save(train_y, f'{self.save_dir}models/train_y.pt')

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

                # sliding window here
                # Set number of non-zero action entries such that num_non_zero * projection_dim = num_train_targets
                num_non_zero = update_y.size(-1) // proj_dim
                self.model.train_inputs = tuple(
                    tri.unsqueeze(-1) if tri.ndimension() == 1 else tri
                    for tri in (update_x[0:num_non_zero * proj_dim], ))
                self.model.train_targets = update_y[0:num_non_zero * proj_dim]
                self.model.actions_op.blocks.data = torch.concat(
                    (self.model.actions_op.blocks.data[1:], torch.randn(
                        (1, 1)).div(math.sqrt(self.model.num_non_zero))))
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

            train_loader = self.generate_dataloaders(train_x=update_x,
                                                     train_y=update_y)

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
