import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from trainers.base_trainer import BaseTrainer, EITrainer, HartmannTrainer


class ExactGPTrainer(BaseTrainer):

    def run_experiment(self):
        train_x, train_y = self.initialize_data()

        while self.task.num_calls < self.max_oracle_calls:
            # Init exact gp model
            model = SingleTaskGP(
                train_x,
                train_y,
                covar_module=gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()),
                likelihood=gpytorch.likelihoods.GaussianLikelihood().to(
                    self.device),
            )
            exact_gp_mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # fit model to data
            fit_gpytorch_mll(exact_gp_mll)

            x_next = self.data_acquisition_iteration(model, train_y)

            # Evaluate candidates
            y_next = self.objective(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            return super().run_experiment()

    def eval(self):
        pass


class HartmannEIExactGPTrainer(ExactGPTrainer, HartmannTrainer, EITrainer):
    pass
