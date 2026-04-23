import math
from dataclasses import dataclass

import torch


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        if math.isnan(self.failure_tolerance):
            self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))


def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


def get_trust_region_bounds(model,
                            X: torch.Tensor,
                            Y: torch.Tensor,
                            length: float,
                            lb: torch.Tensor,
                            ub: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_center = X[Y.argmax(), :].clone()
    weights = _get_trust_region_weights(model=model, dim=X.shape[-1])
    tr_lb = torch.clamp(x_center - weights * length / 2.0, lb, ub)
    tr_ub = torch.clamp(x_center + weights * length / 2.0, lb, ub)
    return tr_lb, tr_ub


def _get_trust_region_weights(model, dim: int) -> torch.Tensor:
    lengthscales = _extract_lengthscales(model)
    if lengthscales is None:
        return torch.ones(dim, device=model.train_inputs[0].device)

    lengthscales = lengthscales.reshape(-1)
    if lengthscales.numel() == 1:
        return torch.ones(dim, device=lengthscales.device, dtype=lengthscales.dtype)

    if lengthscales.numel() != dim:
        return torch.ones(dim, device=lengthscales.device, dtype=lengthscales.dtype)

    if not torch.all(torch.isfinite(lengthscales)) or torch.any(lengthscales <= 0):
        return torch.ones(dim, device=lengthscales.device, dtype=lengthscales.dtype)

    weights = lengthscales / lengthscales.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    return weights


def _extract_lengthscales(model) -> torch.Tensor | None:
    covar_module = model.covar_module

    if hasattr(covar_module, "base_kernel") and hasattr(
            covar_module.base_kernel, "lengthscale"):
        return covar_module.base_kernel.lengthscale.detach().squeeze()

    if hasattr(covar_module, "lengthscale"):
        return covar_module.lengthscale.detach().squeeze()

    return None
