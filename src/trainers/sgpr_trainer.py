import logging

import torch
from models.sgpr import SGPR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from trainers.acquisition_fn_trainers import EITrainer, LogEITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import (GuacamolTrainer, HartmannTrainer,
                                    LassoDNATrainer, LunarTrainer,
                                    RoverTrainer)

import gpytorch
from botorch.fit import fit_gpytorch_mll


class SGPRTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = 'vanilla_sgpr'

    def run_experiment(self, iteration: int):
        # get all attribute information
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
            self.tracker.log({
                'initial y max': train_y.max().item(),
                'best reward': train_y.max().item()
            })

        # get inducing points
        inducing_points = train_x[:self.num_inducing_points]

        # init model
        self.model = SGPR(
            train_x=train_x,
            train_y=train_y,
            inducing_points=inducing_points,
            likelihood=gpytorch.likelihoods.GaussianLikelihood().to(
                self.device),
            kernel_type=self.kernel_type,
            kernel_likelihood_prior=self.kernel_likelihood_prior,
            use_ard_kernel=self.use_ard_kernel).to(self.device, self.data_type)

        # set custom LR on IP and variational parameters
        variational_params_and_ip = [
            p for name, p in self.model.named_parameters()
            if 'variational' in name
        ]
        others = [
            p for name, p in self.model.named_parameters()
            if 'variational' not in name
        ]

        self.optimizer = self.optimizer_type(
            [{
                'params': others
            }, {
                'params': variational_params_and_ip,
                'lr': self.svgp_inducing_point_learning_rate
            }],
            lr=self.learning_rate)

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
                if self.reinit_hyperparams:
                    self.model.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    ).to(self.device)
                    base_kernel = gpytorch.kernels.MaternKernel(
                        2.5, ard_num_dims=None)

                    covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
                    self.model.covar_module = covar_module
                if self.reinit_mean:
                    self.model.mean_module = gpytorch.means.ConstantMean()
            else:
                update_x = train_x
                update_y = model_train_y.squeeze()

            # need this for RAASP sampling
            self.model.train_inputs = tuple(
                tri.unsqueeze(-1) if tri.ndimension() == 1 else tri
                for tri in (update_x, ))

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model)

            if self.optimizer_type == 'botorch_lbfgs':
                self.model.train()
                mll = fit_gpytorch_mll(mll)
                epochs_trained = -1
                final_loss = -1
            else:
                train_loader = self.generate_dataloaders(
                    train_x=update_x,
                    train_y=self.model.outcome_transform(
                        update_y.unsqueeze(1))[0].squeeze())

                final_loss, epochs_trained = self.train_model(
                    train_loader, mll)
            self.model.eval()

            x_next, x_af_val, origin = self.data_acquisition_iteration(
                self.model, model_train_y, train_x)

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
                                   train_rmse=-1,
                                   train_nll=-1,
                                   cos_sim_incum=cos_sim_incum,
                                   epochs_trained=epochs_trained,
                                   x_af_val=x_af_val.item(),
                                   x_next_sigma=x_next_sigma.item(),
                                   standardized_gain=standardized_gain.item(),
                                   candidate_origin=origin)

            reward.append(train_y.max().item())

    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator(device=self.device))
        return train_loader


class OsmbLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='osmb', **kwargs)


class FexoLogEISGPRTrainer(SGPRTrainer, GuacamolTrainer, LogEITrainer):

    def __init__(self, **kwargs):
        super().__init__(molecule='fexo', **kwargs)
