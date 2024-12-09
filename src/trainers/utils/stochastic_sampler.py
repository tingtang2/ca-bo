from botorch.posteriors import Posterior
from torch import Tensor

from trainers.utils.mc_sampler import MCSampler


class StochasticSampler(MCSampler):
    r"""A sampler that simply calls `posterior.rsample` to generate the
    samples. This should only be used for stochastic optimization of the
    acquisition functions, e.g., via `gen_candidates_torch`. This should
    not be used with `optimize_acqf`, which uses deterministic optimizers
    under the hood.

    NOTE: This ignores the `seed` option.
    """

    def forward(self, posterior: Posterior) -> Tensor:
        r"""Draws MC samples from the posterior.

        Args:
            posterior: The posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        return posterior.rsample(sample_shape=self.sample_shape)
