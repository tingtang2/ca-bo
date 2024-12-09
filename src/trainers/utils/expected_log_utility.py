import torch
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

from trainers.utils.safe_math import log_softplus

softplus_func = torch.nn.Softplus()


def get_expected_log_utility_ei(
    model,
    best_f,
    x_next,  # (q,d)
    device,
    use_botorch_stable_log_softplus=False,
):
    output = model(x_next)

    def log_utility(y, ):
        # compute log utility based on y and best_f
        if use_botorch_stable_log_softplus:
            log_utility = log_softplus(y - best_f)
        else:
            log_utility = torch.log(softplus_func(y - best_f))
        return log_utility.to(device)

    ghq = GaussHermiteQuadrature1D()
    ghq = ghq.to(device)
    expected_log_utility = ghq(log_utility, output)

    return expected_log_utility
