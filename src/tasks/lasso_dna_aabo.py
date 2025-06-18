import LassoBench
import numpy as np
import torch

from tasks.objective import Objective


class LassoDNA(Objective):
    """
    https://github.com/ksehic/LassoBench
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.dna_func = LassoBench.RealBenchmark(pick_data='dna')
        super().__init__(
            dim=self.dna_func.n_features,
            lb=-1.0,
            ub=1.0,
            **kwargs,
        )

    def f(self, x):
        self.num_calls += 1
        X_np = x.detach().cpu().numpy().flatten().astype(np.float64)
        y = self.dna_func.evaluate(X_np)
        # negate to create maximization problem
        y = y * -1

        return y


if __name__ == "__main__":
    obj = LassoDNA()
    xs = torch.rand(3, obj.dim) * (obj.ub - obj.lb) + obj.lb
    ys = obj(xs)
    print(xs.shape, ys.shape, ys, obj.num_calls)
