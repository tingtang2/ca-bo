import gpytorch
import torch
from gpytorch import likelihoods
from gpytorch.models.computation_aware_gp import ComputationAwareGP


class CaGP(ComputationAwareGP):

    def __init__(self, train_inputs: torch.Tensor, train_targets: torch.Tensor,
                 likelihood: "likelihoods.GaussianLikelihood",
                 projection_dim: int):
        mean_module = gpytorch.means.ConstantMean()
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

        super(CaGP, self).__init__(train_inputs=train_inputs,
                                   train_targets=train_targets,
                                   mean_module=mean_module,
                                   covar_module=covar_module,
                                   likelihood=likelihood,
                                   projection_dim=projection_dim)
