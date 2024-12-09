import logging

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ComputationAwareELBO
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from models.ca_gp import CaGP
from trainers.acquisition_fn_trainers import EITrainer, LogEITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import HartmannTrainer


class CaGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_inducing_points = 100
        self.grad_clip = 2.0

        self.early_stopping_threshold = 3
        self.train_batch_size = 32

        self.update_train_size = 100

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

            self.model = CaGP(train_inputs=train_x,
                              train_targets=train_y.squeeze(),
                              projection_dim=train_x.size(0) // 2,
                              likelihood=GaussianLikelihood().to(
                                  self.device)).to(self.device)

            self.optimizer = self.optimizer_type(
                [{
                    'params': self.model.parameters(),
                }], lr=self.learning_rate)

            mll = ComputationAwareELBO(self.model.likelihood, self.model)

            train_loader = self.generate_dataloaders(train_x=train_x,
                                                     train_y=train_y.squeeze())

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            x_next = self.data_acquisition_iteration(self.model,
                                                     train_y,
                                                     train_x,
                                                     raw_samples=10)

            # Evaluate candidates
            y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            if not self.turn_off_wandb:
                self.tracker.log({
                    'Num oracle calls':
                    self.task.num_calls - 1,
                    'best reward':
                    train_y.max().item(),
                    'final svgp loss':
                    final_loss,
                    'epochs trained':
                    epochs_trained,
                    'noise param':
                    self.model.likelihood.noise.item()
                })

            logging.info(
                f'Num oracle calls: {self.task.num_calls - 1}, best reward: {train_y.max().item():.3f}, final cagp loss: {final_loss:.3f}, epochs trained: {epochs_trained}'
            )
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
            # logging.info(f'epoch: {i} training loss: {loss:.3f}')

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

    def eval(self):
        pass


class HartmannEICaGPTrainer(CaGPTrainer, HartmannTrainer, EITrainer):
    pass


class HartmannLogEICaGPTrainer(CaGPTrainer, HartmannTrainer, LogEITrainer):
    pass
