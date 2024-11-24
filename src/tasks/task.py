from abc import ABC, abstractmethod

import torch


class Task(ABC):

    def __init__(self, dim: int, lower_bound: torch.Tensor,
                 upper_bound: torch.Tensor):
        super().__init__()

        # number of times oracle has been called
        self.num_calls = 0

        self.dim = dim

        # bounds on domain
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @abstractmethod
    def function_eval(self, x):
        pass
