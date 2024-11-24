import torch
from botorch.test_functions import Hartmann

from tasks.task import Task


class Hartmann6D(Task):

    def __init__(self, device: torch.device):
        self.neg_hartmann6 = Hartmann(negate=True)

        super().__init__(dim=6,
                         upper_bound=1 * torch.ones(6).to(device),
                         lower_bound=0 * torch.ones(6).to(device))

    def function_eval(self, x):
        # known optimum at [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
        self.num_calls += 1
        return self.neg_hartmann6(x)
