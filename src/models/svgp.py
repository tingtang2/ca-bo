# adapted from https://github.com/nataliemaus/aabo/blob/main/svgp/model.py

import gpytorch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  VariationalStrategy)


class SVGPModel(ApproximateGP):

    def __init__(self,
                 inducing_points,
                 likelihood,
                 learn_inducing_locations=True):

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood
        self.num_outputs = 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self,
                  X,
                  output_indices=None,
                  observation_noise=False,
                  *args,
                  **kwargs) -> GPyTorchPosterior:
        self.eval()
        self.likelihood.eval()
        dist = self.likelihood(self(X))

        return GPyTorchPosterior(dist)
