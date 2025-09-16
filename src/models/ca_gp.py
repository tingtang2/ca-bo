import gpytorch
import torch
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    get_matern_kernel_with_gamma_prior)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.transforms.outcome import Standardize
from gpytorch import likelihoods
from gpytorch.models.computation_aware_gp import ComputationAwareGP


class CaGP(ComputationAwareGP):

    def __init__(self,
                 train_inputs: torch.Tensor,
                 train_targets: torch.Tensor,
                 likelihood: "likelihoods.GaussianLikelihood",
                 projection_dim: int,
                 kernel_type: str,
                 init_mode: str,
                 kernel_likelihood_prior: str = None,
                 use_ard_kernel: bool = False,
                 standardize_outputs: bool = False):

        if use_ard_kernel:
            ard_num_dims = train_inputs.shape[-1]
        else:
            ard_num_dims = None

        if kernel_likelihood_prior == 'gamma':
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
        mean_module = gpytorch.means.ConstantMean()
        if standardize_outputs:
            outcome_transform = Standardize(
                m=1, batch_shape=train_inputs.shape[:-2])
            outcome_transform.train()
            train_targets, train_Yvar = outcome_transform(
                Y=train_targets.unsqueeze(1), Yvar=None, X=train_inputs)

        super(CaGP, self).__init__(train_inputs=train_inputs,
                                   train_targets=train_targets.squeeze(),
                                   mean_module=mean_module,
                                   covar_module=covar_module,
                                   likelihood=likelihood,
                                   projection_dim=projection_dim,
                                   initialization=init_mode)

        # need these attributes for BoTorch to work
        self._has_transformed_inputs = False  # need this for RAASP sampling
        self.num_outputs = 1
        self.outcome_transform = None
        if standardize_outputs:
            self.outcome_transform = outcome_transform

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
