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

from torch.utils.data import DataLoader, TensorDataset
from trainers.acquisition_fn_trainers import EITrainer, LogEITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import (GuacamolTrainer, HartmannTrainer,
                                    LassoDNATrainer, LunarTrainer,
                                    RoverTrainer)

from models.exact_gp import ExactGPModel


class ExactGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if not self.turn_off_wandb:
        #     self.tracker.watch(self.model,
        #                        log='all',
        #                        log_freq=20,
        #                        log_graph=True)

        self.name = 'exact_gp'

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
                    base_kernel = gpytorch.kernels.RBFKernel(
                        ard_num_dims=ard_num_dims)
                elif self.kernel_type == 'matern_3_2':
                    base_kernel = gpytorch.kernels.MaternKernel(
                        1.5, ard_num_dims=ard_num_dims)
                else:
                    base_kernel = gpytorch.kernels.MaternKernel(
                        2.5, ard_num_dims=ard_num_dims)

                covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                    self.device)

            assert covar_module.base_kernel.ard_num_dims == ard_num_dims

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
            x_next, x_af_val = self.data_acquisition_iteration(
                self.model, model_train_y, train_x)

            # Evaluate candidates
            y_next = self.task(x_next)
            cos_sim_incum = self.compute_cos_sim_to_incumbent(train_x=train_x,
                                                              train_y=train_y,
                                                              x_next=x_next)
            x_next_mu, x_next_sigma = self.calc_predictive_mean_and_std(
                model=self.model, test_point=x_next)
            standardized_gain = (x_next_mu - torch.max(train_y)) / x_next_sigma

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            self.log_wandb_metrics(train_y=train_y,
                                   y_next=y_next.item(),
                                   train_rmse=train_rmse,
                                   cos_sim_incum=cos_sim_incum,
                                   train_nll=train_nll,
                                   x_af_val=x_af_val.item(),
                                   x_next_sigma=x_next_sigma.item(),
                                   standardized_gain=standardized_gain.item())

            reward.append(train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    # just for debugging purposes
    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset,
                                  batch_size=train_x.shape[0],
                                  shuffle=False)
        return train_loader


class GPyTorchExactGPSlidingWindowTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if not self.turn_off_wandb:
        #     self.tracker.watch(self.model,
        #                        log='all',
        #                        log_freq=20,
        #                        log_graph=True)

        self.name = 'gpytorch_exact_gp_sliding_window'

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

            # only update on recently acquired points
            if i > 0:
                update_x = train_x[-self.update_train_size:]
                # y needs to only have 1 dimension when training in gpytorch
                update_y = model_train_y.squeeze()[-self.update_train_size:]
            else:
                update_x = train_x
                update_y = model_train_y.squeeze()

            # Init exact gp model
            if self.use_ard_kernel:
                ard_num_dims = train_x.shape[-1]
            else:
                ard_num_dims = None

            if self.kernel_likelihood_prior == 'gamma':
                covar_module = get_matern_kernel_with_gamma_prior(
                    ard_num_dims=ard_num_dims)
                likelihood = get_gaussian_likelihood_with_gamma_prior()

                assert covar_module.ard_num_dims == ard_num_dims
            elif self.kernel_likelihood_prior == 'lognormal':
                covar_module = get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=ard_num_dims, use_rbf_kernel=False)
                likelihood = get_gaussian_likelihood_with_lognormal_prior()

                assert covar_module.ard_num_dims == ard_num_dims
            else:
                if self.kernel_type == 'rbf':
                    base_kernel = gpytorch.kernels.RBFKernel(
                        ard_num_dims=ard_num_dims)
                elif self.kernel_type == 'matern_3_2':
                    base_kernel = gpytorch.kernels.MaternKernel(
                        1.5, ard_num_dims=ard_num_dims)
                else:
                    base_kernel = gpytorch.kernels.MaternKernel(
                        2.5, ard_num_dims=ard_num_dims)

                covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                    self.device)

                assert covar_module.base_kernel.ard_num_dims == ard_num_dims

            self.model = ExactGPModel(
                update_x,
                update_y,
                covar_module=covar_module,
                likelihood=likelihood,
            ).to(self.device)

            if self.optimizer_type == torch.optim.Adam:
                self.optimizer = self.optimizer_type(self.model.parameters,
                                                     lr=self.learning_rate)
            elif self.optimizer_type == torch.optim.LBFGS:
                self.optimizer = self.optimizer_type(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    line_search_fn='strong_wolfe')

            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

            # fit model to data
            train_loader = self.generate_dataloaders(train_x=update_x,
                                                     train_y=update_y)

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            # get train rmse
            train_rmse = self.eval(train_x, model_train_y)
            train_nll = self.compute_nll(train_x, model_train_y.squeeze(), mll)
            x_next, x_af_val = self.data_acquisition_iteration(
                self.model, model_train_y, train_x)

            # Evaluate candidates
            y_next = self.task(x_next)
            cos_sim_incum = self.compute_cos_sim_to_incumbent(train_x=train_x,
                                                              train_y=train_y,
                                                              x_next=x_next)

            x_next_mu, x_next_sigma = self.calc_predictive_mean_and_std(
                model=self.model, test_point=x_next)

            standardized_gain = (x_next_mu - torch.max(train_y)) / x_next_sigma

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            self.log_wandb_metrics(train_y=train_y,
                                   y_next=y_next.item(),
                                   train_rmse=train_rmse,
                                   cos_sim_incum=cos_sim_incum,
                                   train_nll=train_nll,
                                   x_af_val=x_af_val.item(),
                                   x_next_sigma=x_next_sigma.item(),
                                   standardized_gain=standardized_gain.item())

            reward.append(train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    # just for debugging purposes
    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset,
                                  batch_size=train_x.shape[0],
                                  shuffle=False)
        return train_loader


class ExactGPSlidingWindowTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if not self.turn_off_wandb:
        #     self.tracker.watch(self.model,
        #                        log='all',
        #                        log_freq=20,
        #                        log_graph=True)

        self.name = 'exact_gp_sliding_window'

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

            # only update on recently acquired points
            if i > 0:
                update_x = train_x[-self.update_train_size:]
                update_y = model_train_y[-self.update_train_size:]
            else:
                update_x = train_x
                update_y = model_train_y

            # Init exact gp model
            if self.use_ard_kernel:
                ard_num_dims = train_x.shape[-1]
            else:
                ard_num_dims = None

            if self.kernel_likelihood_prior == 'gamma':
                covar_module = get_matern_kernel_with_gamma_prior(
                    ard_num_dims=ard_num_dims)
                likelihood = get_gaussian_likelihood_with_gamma_prior()

                assert covar_module.ard_num_dims == ard_num_dims
            elif self.kernel_likelihood_prior == 'lognormal':
                covar_module = get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=ard_num_dims, use_rbf_kernel=False)
                likelihood = get_gaussian_likelihood_with_lognormal_prior()

                assert covar_module.ard_num_dims == ard_num_dims
            else:
                if self.kernel_type == 'rbf':
                    base_kernel = gpytorch.kernels.RBFKernel(
                        ard_num_dims=ard_num_dims)
                elif self.kernel_type == 'matern_3_2':
                    base_kernel = gpytorch.kernels.MaternKernel(
                        1.5, ard_num_dims=ard_num_dims)
                else:
                    base_kernel = gpytorch.kernels.MaternKernel(
                        2.5, ard_num_dims=ard_num_dims)

                covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                    self.device)

                assert covar_module.base_kernel.ard_num_dims == ard_num_dims

            self.model = SingleTaskGP(
                update_x,
                update_y,
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
            x_next, x_af_val = self.data_acquisition_iteration(
                self.model, model_train_y, train_x)

            # Evaluate candidates
            y_next = self.task(x_next)
            cos_sim_incum = self.compute_cos_sim_to_incumbent(train_x=train_x,
                                                              train_y=train_y,
                                                              x_next=x_next)

            x_next_mu, x_next_sigma = self.calc_predictive_mean_and_std(
                model=self.model, test_point=x_next)

            standardized_gain = (x_next_mu - torch.max(train_y)) / x_next_sigma

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            self.log_wandb_metrics(train_y=train_y,
                                   y_next=y_next.item(),
                                   train_rmse=train_rmse,
                                   cos_sim_incum=cos_sim_incum,
                                   train_nll=train_nll,
                                   x_af_val=x_af_val.item(),
                                   x_next_sigma=x_next_sigma.item(),
                                   standardized_gain=standardized_gain.item())

            reward.append(train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    # just for debugging purposes
    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset,
                                  batch_size=train_x.shape[0],
                                  shuffle=False)
        return train_loader


class HartmannEIExactGPTrainer(ExactGPTrainer, HartmannTrainer, EITrainer):
    pass


class LunarEIExactGPTrainer(ExactGPTrainer, LunarTrainer, EITrainer):
    pass


class RoverEIExactGPTrainer(ExactGPTrainer, RoverTrainer, EITrainer):
    pass


class LassoDNALogEIExactGPTrainer(ExactGPTrainer, LassoDNATrainer,
                                  LogEITrainer):
    pass


class OsmbLogEIExactGPTrainer(ExactGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class FexoLogEIExactGPTrainer(ExactGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class Med1LogEIExactGPTrainer(ExactGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med1', **kwargs)


class Med2LogEIExactGPTrainer(ExactGPTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med2', **kwargs)


class RoverEIExactGPSlidingWindowTrainer(ExactGPSlidingWindowTrainer,
                                         RoverTrainer, EITrainer):
    pass


class LassoDNALogEIExactGPSlidingWindowTrainer(ExactGPSlidingWindowTrainer,
                                               LassoDNATrainer, LogEITrainer):
    pass


class OsmbLogEIExactGPSlidingWindowTrainer(ExactGPSlidingWindowTrainer,
                                           GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class FexoLogEIExactGPSlidingWindowTrainer(ExactGPSlidingWindowTrainer,
                                           GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class FexoLogEIGPyTorchExactGPSlidingWindowTrainer(
        GPyTorchExactGPSlidingWindowTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)


class Med1LogEIExactGPSlidingWindowTrainer(ExactGPSlidingWindowTrainer,
                                           GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med1', **kwargs)


class Med2LogEIExactGPSlidingWindowTrainer(ExactGPSlidingWindowTrainer,
                                           GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='med2', **kwargs)
