import gpytorch
from gpytorch.models import ExactGP
from botorch.posteriors.gpytorch import GPyTorchPosterior


class ExactGPModel(ExactGP):

    def __init__(self, train_x, train_y, covar_module, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

        # need these attributes for BoTorch to work
        self._has_transformed_inputs = False  # need this for RAASP sampling
        self.num_outputs = 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, X):
        return X

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
