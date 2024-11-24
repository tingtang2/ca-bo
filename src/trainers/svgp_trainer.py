import logging

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

from models.svgp import SVGPModel
from trainers.acquisition_fn_trainers import EITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import HartmannTrainer

from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader


class SVGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_inducing_points = 100
        self.grad_clip = 2.0
        self.early_stopping_threshold = 10
        self.train_batch_size = 32

    def run_experiment(self, iteration: int):
        train_x, train_y = self.initialize_data()

        # get inducing points
        inducing_points = train_x[:self.num_inducing_points]

        # init model
        self.model = SVGPModel(inducing_points=inducing_points,
                               likelihood=GaussianLikelihood().to(
                                   self.device)).to(self.device)
        print(self.learning_rate)
        self.optimizer = self.optimizer_type(
            [{
                'params': self.model.parameters(),
                # 'params': self.model.likelihood.parameters()
            }],
            lr=self.learning_rate)
        self.mll = VariationalELBO(self.model.likelihood,
                                   self.model,
                                   num_data=train_y.size(0))

        while self.task.num_calls < self.max_oracle_calls:
            if self.norm_data:
                # get normalized train y
                train_y_mean = train_y.mean()
                train_y_std = train_y.std()
                if train_y_std == 0:
                    train_y_std = 1
                train_y = (train_y - train_y_mean) / train_y_std

            train_loader = self.generate_dataloaders(train_x=train_x,
                                                     train_y=train_y)

            self.train_model(train_loader)

            x_next = self.data_acquisition_iteration(self.model, train_y)

            # Evaluate candidates
            y_next = self.task.function_eval(x_next)
            y_next = y_next.unsqueeze(-1)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            logging.info(
                f'Num oracle calls: {self.task.num_calls}, best reward: {train_y.max().item():.3f}'
            )

    def train_model(self, train_loader: DataLoader):
        self.model.train()
        best_loss = 1e+5
        early_stopping_counter = 0
        for i in range(self.epochs):
            loss = self.train_epoch(train_loader)
            # logging.info(f'epoch: {i} training loss: {loss:.3f}')

            if loss < best_loss:
                # self.save_model(f'{self.name}_{iter}')
                early_stopping_counter = 0
                best_loss = loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                break

    def train_epoch(self, train_loader: DataLoader):
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            self.optimizer.zero_grad()

            output = self.model(x.to(self.device))
            loss = -self.mll(output, y.to(self.device))
            loss = loss.sum()

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


class HartmannEISVGPTrainer(SVGPTrainer, HartmannTrainer, EITrainer):
    pass
