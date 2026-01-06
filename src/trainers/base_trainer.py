import copy
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import torch
from functions.LBFGS import FullBatchLBFGS
from tasks.task import Task
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader

import gpytorch
from gpytorch.metrics import mean_squared_error


class BaseTrainer(ABC):

    def __init__(self,
                 optimizer_type,
                 device: str,
                 save_dir: Union[str, Path],
                 batch_size: int,
                 dropout_prob: float,
                 learning_rate: float,
                 max_oracle_calls: int,
                 save_plots: bool = True,
                 seed: int = 11202022,
                 norm_data: bool = False,
                 tracker=None,
                 debug: bool = False,
                 **kwargs) -> None:
        super().__init__()

        # basic configs every trainer needs
        self.optimizer_type = optimizer_type
        self.device = torch.device(device)
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.seed = seed
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.debug = debug

        # BO specific
        self.task: Task = None
        self.max_oracle_calls = max_oracle_calls
        self.norm_data = norm_data

        # wandb tracking
        self.tracker = tracker

        # extra configs in form of kwargs
        for key, item in kwargs.items():
            setattr(self, key, item)

    @abstractmethod
    def data_acquisition_iteration(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def run_experiment(self, iteration: int):
        pass

    @abstractmethod
    def initialize_data(self):
        pass

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), f'{self.save_dir}models/{name}.pt')

    def save_metrics(self, metrics: List[float], iter: int, name: str):
        save_name = f'{name}_iteration_{iter}-{datetime.now().strftime("%m_%d_%Y_%H:%M:%S")}.json'
        with open(Path(Path.home(), self.save_dir, 'metrics/', save_name),
                  'w') as f:
            json.dump(metrics, f)

    def init_new_run(self, tracker):
        self.tracker = tracker

    def compute_nll(self, x, y, exact_mll):
        with torch.no_grad():
            output = self.model(x.to(self.device))
        return -exact_mll(output, y.to(self.device)).mean().item()

    def eval(self, train_x, train_y):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(train_x)
        return mean_squared_error(preds,
                                  train_y.to(self.device),
                                  squared=False).mean().item()

    def compute_cos_sim_to_incumbent(self, train_x, train_y, x_next):
        incumbent = train_x[torch.argmax(train_y)]

        return cosine_similarity(incumbent, x_next).item()

    def calc_log_det_kernel_ips(self):
        self.model.eval()

        # get inducing points
        if 'svgp' in self.name:
            ips = self.model.variational_strategy.inducing_points
        else:
            ips = self.model.covar_module.inducing_points

        # get K(Z, Z)
        covar_ips_lazy = self.model.covar_module(ips, ips)

        return covar_ips_lazy.logdet().item()

    def calc_cond_num_SKS(self):
        # reconstitute S^T K^hat S matrix from cholesky factor
        gram_SKhatS = self.model.cholfac_gram_SKhatS @ self.model.cholfac_gram_SKhatS.T

        return torch.linalg.cond(gram_SKhatS).item()

    def calc_predictive_mean_and_std(self, model, test_point):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model.eval()

            posterior = model(test_point)

            mu = posterior.mean
            sigma = posterior.stddev

        return mu, sigma

    def log_wandb_metrics(self,
                          train_y: torch.Tensor,
                          final_loss: float = -1,
                          epochs_trained: int = -1,
                          train_rmse: float = -1.0,
                          train_nll: float = -1.0,
                          log_to_file: bool = True,
                          y_next: float = -1.0,
                          cos_sim_incum: float = -1.0,
                          action_norm: float = -1.0,
                          x_af_val: float = -1.0,
                          x_next_sigma: float = -1.0,
                          standardized_gain: float = -1.0,
                          candidate_origin: int = -1):

        passed_model = self.model

        if self.kernel_likelihood_prior == 'lognormal' or (
                self.kernel_type == 'spherical_linear'
                and not self.use_output_scale):
            outputscale = torch.tensor([1])
            raw_lengthscale = passed_model.covar_module.raw_lengthscale
            constraint = passed_model.covar_module.raw_lengthscale_constraint
            lengthscale = constraint.transform(raw_lengthscale)
        elif 'sgpr' in self.name:
            raw_outputscale = passed_model.covar_module.base_kernel.raw_outputscale
            constraint = passed_model.covar_module.base_kernel.raw_outputscale_constraint
            outputscale = constraint.transform(raw_outputscale)

            raw_lengthscale = passed_model.covar_module.base_kernel.base_kernel.raw_lengthscale
            constraint = passed_model.covar_module.base_kernel.base_kernel.raw_lengthscale_constraint
            lengthscale = constraint.transform(raw_lengthscale)

        else:
            raw_outputscale = passed_model.covar_module.raw_outputscale
            constraint = passed_model.covar_module.raw_outputscale_constraint
            outputscale = constraint.transform(raw_outputscale)

            raw_lengthscale = passed_model.covar_module.base_kernel.raw_lengthscale
            constraint = passed_model.covar_module.base_kernel.raw_lengthscale_constraint
            lengthscale = constraint.transform(raw_lengthscale)

        if 'exact' in self.trainer_type:
            log_dict = {
                'Num oracle calls':
                self.task.num_calls - 1,
                'best reward':
                train_y.max().item(),
                'noise param':
                passed_model.likelihood.noise.item(),
                'lengthscale param':
                torch.mean(lengthscale).item()
                if self.use_ard_kernel else lengthscale.item(),
                'outputscale param':
                outputscale.item(),
                'train rmse':
                train_rmse,
                'train nll':
                train_nll,
                'y_next':
                y_next,
                'cos_sim_incum':
                cos_sim_incum,
                'x_af_val':
                x_af_val,
                'x_next_sigma':
                x_next_sigma,
                'standardized_gain':
                standardized_gain,
                'candidate_origin':
                candidate_origin
            }
        elif ('svgp' in self.trainer_type
              or 'sgpr' in self.trainer_type) and self.log_diagnostics:
            log_dict = {
                'Num oracle calls':
                self.task.num_calls - 1,
                'best reward':
                train_y.max().item(),
                'final svgp loss':
                final_loss,
                'epochs trained':
                epochs_trained,
                'noise param':
                self.model.likelihood.noise.item(),
                'lengthscale param':
                torch.mean(lengthscale).item()
                if self.use_ard_kernel else lengthscale.item(),
                'outputscale param':
                outputscale.item(),
                'train rmse':
                train_rmse,
                'train nll':
                train_nll,
                'y_next':
                y_next,
                'cos_sim_incum':
                cos_sim_incum,
                'action_norm':
                action_norm,
                # 'log det K(z, z)':
                # self.calc_log_det_kernel_ips(),
                'log det K(z, z)':
                -1,
                'x_af_val':
                x_af_val,
                'x_next_sigma':
                x_next_sigma,
                'standardized_gain':
                standardized_gain,
                'candidate_origin':
                candidate_origin
            }
        elif 'ca_gp' in self.trainer_type and self.log_diagnostics:
            log_dict = {
                'Num oracle calls':
                self.task.num_calls - 1,
                'best reward':
                train_y.max().item(),
                'final svgp loss':
                final_loss,
                'epochs trained':
                epochs_trained,
                'noise param':
                self.model.likelihood.noise.item(),
                'lengthscale param':
                torch.mean(lengthscale).item()
                if self.use_ard_kernel else lengthscale.item(),
                'outputscale param':
                outputscale.item(),
                'train rmse':
                train_rmse,
                'train nll':
                train_nll,
                'y_next':
                y_next,
                'cos_sim_incum':
                cos_sim_incum,
                'action_norm':
                action_norm,
                # 'S^TK^hatS condition number':
                # self.calc_cond_num_SKS(),
                'S^TK^hatS condition number':
                -1,
                'x_af_val':
                x_af_val,
                'x_next_sigma':
                x_next_sigma,
                'standardized_gain':
                standardized_gain,
                'candidate_origin':
                candidate_origin
            }
        else:
            log_dict = {
                'Num oracle calls':
                self.task.num_calls - 1,
                'best reward':
                train_y.max().item(),
                'final svgp loss':
                final_loss,
                'epochs trained':
                epochs_trained,
                'noise param':
                self.model.likelihood.noise.item(),
                'lengthscale param':
                torch.mean(lengthscale).item()
                if self.use_ard_kernel else lengthscale.item(),
                'outputscale param':
                outputscale.item(),
                'train rmse':
                train_rmse,
                'train nll':
                train_nll,
                'y_next':
                y_next,
                'cos_sim_incum':
                cos_sim_incum,
                'action_norm':
                action_norm,
                'x_af_val':
                x_af_val,
                'x_next_sigma':
                x_next_sigma,
                'standardized_gain':
                standardized_gain,
                'candidate_origin':
                candidate_origin
            }

        if not self.turn_off_wandb:
            self.tracker.log(log_dict)

        if log_to_file:
            logging.info(', '.join(
                [f'{key}: {value:.5f}' for key, value in log_dict.items()]))

    def train_epoch(self, train_loader: DataLoader, mll):
        running_loss = 0.0
        running_kl = 0.0
        running_ll = 0.0
        for i, (x, y) in enumerate(train_loader):
            self.optimizer.zero_grad()

            output = self.model(x.to(self.device))

            if 'ca_gp' in self.name:
                loss, ll, kl = mll(output, y.to(self.device))
                assert kl.item() >= 0
                loss = -loss
                running_kl += kl.item()
                running_ll += ll.item()
            else:
                loss = -mll(output, y.to(self.device))

            loss.backward()
            if self.grad_clip != -1.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.grad_clip)

            self.optimizer.step()

            running_loss += loss.item()

        if self.debug and 'ca_gp' in self.name:
            print(
                f'positive kl (want to min): {running_kl:.3f}, positive ll (want to max): {running_ll:.3f}'
            )
        return running_loss

    def train_epoch_lbfgs(self, train_loader: DataLoader, mll):
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):

            def closure():
                self.optimizer.zero_grad()

                output = self.model(x.to(self.device))
                if 'ca_gp' in self.name:
                    loss, ll, kl = mll(output, y.to(self.device))
                    assert kl.item() >= 0
                    loss = -loss
                else:
                    loss = -mll(output, y.to(self.device))

                loss.backward()
                if self.grad_clip != -1.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.grad_clip)
                return loss

            output = self.model(x.to(self.device))
            if 'ca_gp' in self.name:
                loss, ll, kl = mll(output, y.to(self.device))
                assert kl.item() >= 0
                loss = -loss
            else:
                loss = -mll(output, y.to(self.device))
            running_loss += -loss.item()

            self.optimizer.step(closure)

        return running_loss

    def train_epoch_custom_lbfgs(self, train_loader: DataLoader, mll):
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):

            def closure():
                self.optimizer.zero_grad()

                output = self.model(x.to(self.device))
                if 'ca_gp' in self.name:
                    loss, ll, kl = mll(output, y.to(self.device))
                    assert kl.item() >= 0
                    loss = -loss
                else:
                    loss = -mll(output, y.to(self.device))

                return loss

            loss = closure()
            loss.backward()

            # perform step and update curvature
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, lr, _, F_eval, G_eval, _, _ = self.optimizer.step(options)
            running_loss += loss.item()

        return running_loss

    def train_model(self, train_loader: DataLoader, mll):
        self.model.train()
        best_loss = torch.inf
        early_stopping_counter = 0
        best_model_state = None

        # Only initialize if debugging
        if self.debug:
            grad_norm_history = {}

        for i in range(self.epochs):
            if isinstance(self.optimizer, torch.optim.LBFGS):
                loss = self.train_epoch_lbfgs(train_loader, mll)
            elif isinstance(self.optimizer, FullBatchLBFGS):
                loss = self.train_epoch_custom_lbfgs(train_loader, mll)
            else:
                loss = self.train_epoch(train_loader, mll)

            if loss < best_loss:
                # self.save_model(f'{self.name}')
                best_model_state = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
                best_loss = loss
            else:
                early_stopping_counter += 1

            if self.debug:
                # get hyperparameters
                raw_outputscale = self.model.covar_module.raw_outputscale
                constraint = self.model.covar_module.raw_outputscale_constraint
                outputscale = constraint.transform(raw_outputscale)

                raw_lengthscale = self.model.covar_module.base_kernel.raw_lengthscale
                constraint = self.model.covar_module.base_kernel.raw_lengthscale_constraint
                lengthscale = constraint.transform(raw_lengthscale)

                print(
                    f'epoch: {i} training loss: {loss:.3f} patience: {early_stopping_counter} outputscale: {outputscale.item():.3f} lengthscale: {lengthscale.item():.3f} noise: {self.model.likelihood.noise.item():.3f}'
                )

                # Calculate and store gradient norms
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.detach().data.norm(2).item()
                        print(f'{name} norm: {norm:.6f}')
                        if name not in grad_norm_history:
                            grad_norm_history[name] = []
                        grad_norm_history[name].append(norm)

            if early_stopping_counter == self.early_stopping_threshold:
                # Plot gradient norms before returning
                if self.debug:
                    self._plot_grad_norms(grad_norm_history)
                # Load the best model weights before returning
                self.model.load_state_dict(best_model_state)
                return loss, i + 1

        # Plot gradient norms before returning
        if self.debug:
            self._plot_grad_norms(grad_norm_history)
        self.model.load_state_dict(best_model_state)
        return loss, i + 1

    def _plot_grad_norms(self, grad_norm_history):
        plt.figure(figsize=(10, 6))
        for name, norms in grad_norm_history.items():
            plt.plot(norms, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm (L2)')
        plt.title('Gradient Norms per Parameter over Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.name}_.png')

    def sobol_initialize_data(self) -> Tuple[torch.tensor, torch.tensor]:
        sobol = torch.quasirandom.SobolEngine(dimension=self.task.dim,
                                              scramble=True,
                                              seed=self.seed)
        sobol_draws = sobol.draw(n=self.num_initial_points).to(self.device)
        if self.turn_on_simple_input_transform:
            init_train_x = sobol_draws
        else:
            init_train_x = sobol_draws * (self.task.ub -
                                          self.task.lb) + self.task.lb
        init_train_y = self.task(sobol_draws * (self.task.ub - self.task.lb) +
                                 self.task.lb)

        return init_train_x, init_train_y
