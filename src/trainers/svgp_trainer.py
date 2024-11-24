import logging

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

from models.svgp import SVGPModel
from trainers.acquisition_fn_trainers import EITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import HartmannTrainer


class SVGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_inducing_points = 100

    def run_experiment(self, iteration: int):
        train_x, train_y = self.initialize_data()

        # get inducing points
        inducing_points = train_x[:self.num_inducing_points]

        # init model
        self.model = SVGPModel(inducing_points=inducing_points,
                               likelihood=GaussianLikelihood().to(
                                   self.device)).to(self.device)

        while self.task.num_calls < self.max_oracle_calls:
            if self.norm_data:
                # get normalized train y
                train_y_mean = train_y.mean()
                train_y_std = train_y.std()
                if train_y_std == 0:
                    train_y_std = 1
                train_y = (train_y - train_y_mean) / train_y_std

            self.train_model(train_y=train_y)

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

    def train_model(self, train_y):
        self.model.train()
        optimizer = torch.optim.Adam([{
            'params': self.model.parameters(),
            'lr': self.learning_rate
        }],
                                     lr=self.learning_rate)
        mll = VariationalELBO(self.model.likelihood,
                              self.model,
                              num_data=train_y.size(0))

    def train_epoch(self):
        pass


class HartmannEISVGPTrainer(SVGPTrainer, HartmannTrainer, EITrainer):
    pass
