from gpytorch.means import ConstantMean
from gpytorch.kernels import InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
import gpytorch
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    get_matern_kernel_with_gamma_prior)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.transforms.outcome import Standardize


class SGPR(ExactGP):

    def __init__(self,
                 train_x,
                 train_y,
                 inducing_points,
                 likelihood,
                 kernel_likelihood_prior: str = None,
                 kernel_type: str = 'matern_5_2',
                 use_ard_kernel: bool = False,
                 standardize_outputs: bool = False):
        if use_ard_kernel:
            ard_num_dims = inducing_points.shape[-1]
        else:
            ard_num_dims = None

        if kernel_likelihood_prior == 'gamma':
            self.base_covar_module = get_matern_kernel_with_gamma_prior(
                ard_num_dims=ard_num_dims)
            likelihood = get_gaussian_likelihood_with_gamma_prior()
        elif kernel_likelihood_prior == 'lognormal':
            self.base_covar_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=ard_num_dims, use_rbf_kernel=False)
            likelihood = get_gaussian_likelihood_with_lognormal_prior()
        else:
            if kernel_type == 'rbf':
                base_kernel = gpytorch.kernels.RBFKernel(
                    ard_num_dims=ard_num_dims)
            elif kernel_type == 'matern_3_2':
                base_kernel = gpytorch.kernels.MaternKernel(
                    1.5, ard_num_dims=ard_num_dims)
            else:
                base_kernel = gpytorch.kernels.MaternKernel(
                    2.5, ard_num_dims=ard_num_dims)

            self.base_covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.mean_module = ConstantMean()
        self.covar_module = InducingPointKernel(
            self.base_covar_module,
            inducing_points=inducing_points,
            likelihood=likelihood)

        if standardize_outputs:
            outcome_transform = Standardize(m=1,
                                            batch_shape=train_x.shape[:-2])
            outcome_transform.train()
            train_y, train_Yvar = outcome_transform(Y=train_y.unsqueeze(1),
                                                    Yvar=None,
                                                    X=train_x)
        super(SGPR, self).__init__(train_x, train_y, likelihood)

        # need these attributes for BoTorch to work
        self._has_transformed_inputs = False  # need this for RAASP sampling
        self.num_outputs = 1

        if standardize_outputs:
            self.outcome_transform = outcome_transform

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(self,
                  X,
                  output_indices=None,
                  observation_noise=False,
                  *args,
                  **kwargs) -> GPyTorchPosterior:
        self.eval()
        dist = self(X)
        posterior = GPyTorchPosterior(dist)
        if self.outcome_transform:
            posterior = self.outcome_transform.untransform_posterior(posterior,
                                                                     X=X)

        return posterior
