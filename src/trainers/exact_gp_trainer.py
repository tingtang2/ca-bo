import logging

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    get_matern_kernel_with_gamma_prior)
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import trange

from trainers.acquisition_fn_trainers import EITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import HartmannTrainer, LunarTrainer, RoverTrainer


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
        self.train_y_mean = train_y.mean()
        self.train_y_std = train_y.std()
        if self.train_y_std == 0:
            self.train_y_std = 1

        # log initial y_max
        print(f'initial y max: {train_y.max().item()}')
        logging.info(f'initial y max: {train_y.max().item()}')
        if not self.turn_off_wandb:
            self.tracker.log({'initial y max': train_y.max().item()})

        reward = []

        for i in trange(self.max_oracle_calls - self.num_initial_points):
            if self.norm_data:
                # get normalized train y
                model_train_y = (train_y -
                                 self.train_y_mean) / self.train_y_std
            else:
                model_train_y = train_y

            # Init exact gp model
            if self.use_ard_kernel:
                ard_num_dims = train_x.shape[-1]
            else:
                ard_num_dims = None

            if self.kernel_likelihood_prior == 'gamma':
                covar_module = get_matern_kernel_with_gamma_prior(
                    ard_num_dims=ard_num_dims)
                likelihood = get_gaussian_likelihood_with_gamma_prior()
            elif self.kernel_likelihood_prior == 'lognormal':
                covar_module = get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=ard_num_dims, use_rbf_kernel=False)
                likelihood = get_gaussian_likelihood_with_lognormal_prior()
            else:
                if self.kernel_type == 'rbf':
                    base_kernel = gpytorch.kernels.RBFKernel()
                elif self.kernel_type == 'matern_3_2':
                    base_kernel = gpytorch.kernels.MaternKernel(1.5)
                else:
                    base_kernel = gpytorch.kernels.MaternKernel(2.5)

                covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                    self.device)

            self.model = SingleTaskGP(
                train_x,
                model_train_y,
                covar_module=covar_module,
                likelihood=likelihood,
            ).to(self.device)
            exact_gp_mll = ExactMarginalLogLikelihood(self.model.likelihood,
                                                      self.model)

            # fit model to data
            mll = fit_gpytorch_mll(exact_gp_mll)
            self.model.eval()

            # get train rmse
            train_rmse = self.eval(train_x, model_train_y)
            train_nll = self.compute_nll(train_x, model_train_y.squeeze(), mll)
            x_next = self.data_acquisition_iteration(self.model, model_train_y,
                                                     train_x).to(self.device)

            # Evaluate candidates
            y_next = self.task(x_next)
            cos_sim_incum = self.compute_cos_sim_to_incumbent(train_x=train_x,
                                                              train_y=train_y,
                                                              x_next=x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            self.log_wandb_metrics(train_y=train_y,
                                   y_next=y_next.item(),
                                   train_rmse=train_rmse,
                                   cos_sim_incum=cos_sim_incum,
                                   train_nll=train_nll)

            reward.append(train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)


class HartmannEIExactGPTrainer(ExactGPTrainer, HartmannTrainer, EITrainer):
    pass


class LunarEIExactGPTrainer(ExactGPTrainer, LunarTrainer, EITrainer):
    pass


class RoverEIExactGPTrainer(ExactGPTrainer, RoverTrainer, EITrainer):
    pass
