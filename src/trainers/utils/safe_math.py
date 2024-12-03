import torch
from torch import Tensor

from typing import Union

TAU = 1.0  # default temperature parameter for smooth approximations to non-linearities
from torch.nn.functional import softplus


def log_softplus(x: Tensor, tau: Union[float, Tensor] = TAU) -> Tensor:
    """Computes the logarithm of the softplus function with high numerical accuracy.

    Args:
        x: Input tensor, should have single or double precision floats.
        tau: Decreasing tau increases the tightness of the
            approximation to ReLU. Non-negative and defaults to 1.0.

    Returns:
        Tensor corresponding to `log(softplus(x))`.
    """
    tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device)
    # cutoff chosen to achieve accuracy to machine epsilon
    upper = 16 if x.dtype == torch.float32 else 32
    lower = -15 if x.dtype == torch.float32 else -35
    mask = x / tau > lower
    return torch.where(
        mask,
        softplus(x.masked_fill(~mask, lower), beta=(1 / tau),
                 threshold=upper).log(),
        x / tau + tau.log(),
    )
