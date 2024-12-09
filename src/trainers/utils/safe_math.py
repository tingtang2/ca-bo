import math
from typing import Union

import torch
from torch import Tensor

TAU = 1.0  # default temperature parameter for smooth approximations to non-linearities
_log2 = math.log(2)
from torch.nn.functional import softplus

from trainers.utils.constants import get_constants_like


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


def log1mexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    log2 = get_constants_like(values=_log2, ref=x)
    is_small = -log2 < x  # x < 0
    return torch.where(
        is_small,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )
