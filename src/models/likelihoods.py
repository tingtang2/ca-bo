import torch
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior

MIN_INFERRED_NOISE_LEVEL = 1e-4


def get_gaussian_likelihood_with_gamma_prior(
        batch_shape: torch.Size | None = None) -> GaussianLikelihood:
    """Gaussian likelihood with Gamma(1.1, 0.05) prior and softplus constraint."""
    batch_shape = torch.Size() if batch_shape is None else batch_shape
    noise_prior = GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    return GaussianLikelihood(noise_prior=noise_prior,
                              batch_shape=batch_shape,
                              noise_constraint=GreaterThan(
                                  MIN_INFERRED_NOISE_LEVEL,
                                  initial_value=noise_prior_mode))


def get_gaussian_likelihood_with_lognormal_prior(
        batch_shape: torch.Size | None = None) -> GaussianLikelihood:
    """Gaussian likelihood with LogNormal(-4.0, 1.0) prior and softplus constraint."""
    batch_shape = torch.Size() if batch_shape is None else batch_shape
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    return GaussianLikelihood(noise_prior=noise_prior,
                              batch_shape=batch_shape,
                              noise_constraint=GreaterThan(
                                  MIN_INFERRED_NOISE_LEVEL,
                                  initial_value=noise_prior.mode))
