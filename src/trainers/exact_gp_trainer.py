import logging

import gpytorch
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import HartmannTrainer
from trainers.acquisition_fn_trainers import EITrainer


class ExactGPTrainer(BaseTrainer):

    def run_experiment(self, iteration: int):
        train_x, train_y = self.initialize_data()

        while self.task.num_calls < self.max_oracle_calls:
            if self.norm_data:
                # get normalized train y
                train_y_mean = train_y.mean()
                train_y_std = train_y.std()
                if train_y_std == 0:
                    train_y_std = 1
                train_y = (train_y - train_y_mean) / train_y_std

            # Init exact gp model
            model = SingleTaskGP(
                train_x,
                train_y,
                covar_module=gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()),
                likelihood=gpytorch.likelihoods.GaussianLikelihood().to(
                    self.device),
            ).to(self.device)
            exact_gp_mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # fit model to data
            fit_gpytorch_model(exact_gp_mll)

            x_next = self.data_acquisition_iteration(model, train_y)

            # Evaluate candidates
            y_next = self.task.function_eval(x_next)
            y_next = y_next.unsqueeze(-1)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            logging.info(
                f'Num oracle calls: {self.task.num_calls}, best reward: {train_y.max().item():.3f}'
            )

    def eval(self):
        pass


class HartmannEIExactGPTrainer(ExactGPTrainer, HartmannTrainer, EITrainer):
    pass
