import logging

import gpytorch
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import trange

from trainers.acquisition_fn_trainers import EITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import HartmannTrainer


class ExactGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if not self.turn_off_wandb:
        #     self.tracker.watch(self.model,
        #                        criterion=self.criterion,
        #                        log='all',
        #                        log_freq=20,
        #                        log_graph=True)

    def run_experiment(self, iteration: int):
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()

        reward = []
        if self.kernel_type == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel()
        elif self.kernel_type == 'matern_3_2':
            base_kernel = gpytorch.kernels.MaternKernel(1.5)
        else:
            base_kernel = gpytorch.kernels.MaternKernel(2.5)

        for i in trange(self.max_oracle_calls - self.num_initial_points):
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
                covar_module=gpytorch.kernels.ScaleKernel(base_kernel),
                likelihood=gpytorch.likelihoods.GaussianLikelihood().to(
                    self.device),
            ).to(self.device)
            exact_gp_mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # fit model to data
            fit_gpytorch_model(exact_gp_mll)

            x_next = self.data_acquisition_iteration(model, train_y, train_x)

            # Evaluate candidates
            y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            logging.info(
                f'Num oracle calls: {self.task.num_calls - 1}, best reward: {train_y.max().item():.3f}'
            )

            if not self.turn_off_wandb:
                self.tracker.log({
                    'Num oracle calls': self.task.num_calls - 1,
                    'best reward': train_y.max().item()
                })
            reward.append(train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    def eval(self):
        pass


class HartmannEIExactGPTrainer(ExactGPTrainer, HartmannTrainer, EITrainer):
    pass
