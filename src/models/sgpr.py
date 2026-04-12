import gpytorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    get_matern_kernel_with_gamma_prior)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import InducingPointKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from models.kernels.spherical_linear import SphericalLinearKernel
from models.likelihoods import get_gaussian_likelihood_with_lognormal_prior as custom_get_gaussian_likeliood_with_lognormal_prior


class SGPR(ExactGP):
    def __init__(self,
                 train_x,
                 train_y,
                 inducing_points,
                 likelihood,
                 kernel_likelihood_prior: str = None,
                 kernel_type: str = 'matern_5_2',
                 use_ard_kernel: bool = False,
                 standardize_outputs: bool = False,
                 add_likelihood: bool = False,
                 turn_off_prior: bool = False,
                 ln_noise_prior_loc: float = -4.0,
                 spherical_linear_lengthscale_prior: str = 'dsp_unscaled'):

        self.add_likelihood = add_likelihood
        if standardize_outputs:
            outcome_transform = Standardize(m=1,
                                            batch_shape=train_x.shape[:-2])
            outcome_transform.train()
            train_y, train_Yvar = outcome_transform(Y=train_y.unsqueeze(1),
                                                    Yvar=None,
                                                    X=train_x)
        if use_ard_kernel:
            ard_num_dims = inducing_points.shape[-1]
        else:
            ard_num_dims = None

        selected_likelihood = likelihood
        if kernel_type == 'spherical_linear':
            base_covar_module = SphericalLinearKernel(
                data_dims=inducing_points.shape[-1],
                ard_num_dims=ard_num_dims,
                prior=spherical_linear_lengthscale_prior,
                enable_constraint_transform=True,
                turn_off_prior=turn_off_prior)
            if turn_off_prior:
                selected_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            else:
                selected_likelihood = custom_get_gaussian_likeliood_with_lognormal_prior(
                    loc=ln_noise_prior_loc)
        elif kernel_likelihood_prior == 'gamma':
            base_covar_module = get_matern_kernel_with_gamma_prior(
                ard_num_dims=ard_num_dims)
            selected_likelihood = get_gaussian_likelihood_with_gamma_prior()
        elif kernel_likelihood_prior == 'lognormal':
            base_covar_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=ard_num_dims, use_rbf_kernel=False)
            selected_likelihood = get_gaussian_likelihood_with_lognormal_prior()
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

            base_covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        super(SGPR, self).__init__(train_x, train_y.squeeze(),
                                   selected_likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = InducingPointKernel(
            base_covar_module,
            inducing_points=inducing_points,
            likelihood=self.likelihood)

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
        if self.add_likelihood:
            dist = self.likelihood(self(X))
        else:
            dist = self(X)
        posterior = GPyTorchPosterior(dist)
        if self.outcome_transform:
            posterior = self.outcome_transform.untransform_posterior(posterior,
                                                                     X=X)

        return posterior

    def transform_inputs(self, X):
        return X
