import copy
import logging

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.metrics import mean_squared_error
from gpytorch.mlls import ComputationAwareELBO
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from models.ca_gp import CaGP
from trainers.acquisition_fn_trainers import EITrainer, LogEITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import HartmannTrainer, LunarTrainer
from trainers.svgp_trainer import SVGPEULBOTrainer


class CaGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_inducing_points = 100
        self.grad_clip = 2.0

        self.train_batch_size = 32

        self.update_train_size = 100

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()
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

            self.model = CaGP(train_inputs=train_x,
                              train_targets=model_train_y.squeeze(),
                              projection_dim=int(self.proj_dim_ratio *
                                                 train_x.size(0)),
                              likelihood=GaussianLikelihood().to(self.device),
                              kernel_type=self.kernel_type,
                              init_mode=self.ca_gp_init_mode).to(self.device)

            self.optimizer = self.optimizer_type(
                [{
                    'params': self.model.parameters(),
                }], lr=self.learning_rate)

            mll = ComputationAwareELBO(self.model.likelihood, self.model)

            train_loader = self.generate_dataloaders(
                train_x=train_x, train_y=model_train_y.squeeze())

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            train_rmse = self.eval(train_x, train_y)

            x_next = self.data_acquisition_iteration(self.model, model_train_y,
                                                     train_x).to(self.device)

            # Evaluate candidates
            y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            self.log_wandb_metrics(train_y=train_y,
                                   final_loss=final_loss,
                                   train_rmse=train_rmse,
                                   epochs_trained=epochs_trained)

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
                # self.save_model(f'{self.name}_{iter}')
                best_model_state = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
                best_loss = loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                # Load the best model weights before returning
                self.model.load_state_dict(best_model_state)
                return loss, i + 1

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

    def eval(self, train_x, train_y):
        self.model.eval()
        preds = self.model(train_x)
        return mean_squared_error(preds,
                                  train_y.to(self.device),
                                  squared=False).mean().item()


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

            self.model = CaGP(train_inputs=train_x,
                              train_targets=model_train_y.squeeze(),
                              projection_dim=int(self.proj_dim_ratio *
                                                 train_x.size(0)),
                              likelihood=GaussianLikelihood().to(self.device),
                              kernel_type=self.kernel_type,
                              init_mode=self.ca_gp_init_mode).to(self.device)

            self.optimizer = self.optimizer_type(
                [{
                    'params': self.model.parameters(),
                }], lr=self.learning_rate)

            mll = ComputationAwareELBO(self.model.likelihood, self.model)

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

            train_rmse = self.eval(train_x, train_y)

            # Evaluate candidates
            y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            self.log_wandb_metrics(train_y=train_y,
                                   train_rmse=train_rmse,
                                   final_loss=final_loss,
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


class HartmannEICaGPTrainer(CaGPTrainer, HartmannTrainer, EITrainer):
    pass


class HartmannLogEICaGPTrainer(CaGPTrainer, HartmannTrainer, LogEITrainer):
    pass


class HartmannEICaGPEULBOTrainer(CaGPEULBOTrainer, HartmannTrainer, EITrainer):
    pass


class LunarEICaGPTrainer(CaGPTrainer, LunarTrainer, EITrainer):
    pass


class LunarEICaGPEULBOTrainer(CaGPEULBOTrainer, LunarTrainer, EITrainer):
    pass
