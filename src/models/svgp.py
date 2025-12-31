# adapted from https://github.com/nataliemaus/aabo/blob/main/svgp/model.py

import gpytorch
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    get_matern_kernel_with_gamma_prior)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  VariationalStrategy)
from models.kernels.spherical_linear import SphericalLinearKernel


class SVGPModel(ApproximateGP):

    def __init__(self,
                 inducing_points,
                 likelihood,
                 learn_inducing_locations=True,
                 kernel_type: str = 'matern_5_2',
                 kernel_likelihood_prior: str = None,
                 use_ard_kernel: bool = False,
                 standardize_outputs: bool = False,
                 add_likelihood: bool = False):

        self.add_likelihood = add_likelihood

        if use_ard_kernel:
            ard_num_dims = inducing_points.shape[-1]
        else:
            ard_num_dims = None

        if kernel_type == 'spherical_linear':
            covar_module = SphericalLinearKernel(ard_num_dims=ard_num_dims)
            likelihood = get_gaussian_likelihood_with_lognormal_prior()
        elif kernel_likelihood_prior == 'gamma':
            covar_module = get_matern_kernel_with_gamma_prior(
                ard_num_dims=ard_num_dims)
            likelihood = get_gaussian_likelihood_with_gamma_prior()
        elif kernel_likelihood_prior == 'lognormal':
            covar_module = get_covar_module_with_dim_scaled_prior(
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

            covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

            assert covar_module.base_kernel.ard_num_dims == ard_num_dims

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
        self.covar_module = covar_module
        self.likelihood = likelihood

        # need these attributes for BoTorch to work
        self._has_transformed_inputs = False  # need this for RAASP sampling
        self.num_outputs = 1
        self.outcome_transform = None

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
        if self.add_likelihood:
            dist = self.likelihood(self(X))
        else:
            dist = self(X)
        posterior = GPyTorchPosterior(dist)
        if self.outcome_transform:
            posterior = self.outcome_transform.untransform_posterior(posterior,
                                                                     X=X)

        return posterior
