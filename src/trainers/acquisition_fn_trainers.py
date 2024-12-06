import copy

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

from trainers.base_trainer import BaseTrainer


class EITrainer(BaseTrainer):

    def __init__(self, raw_samples: int = 6, num_restarts: int = 6, **kwargs):
        super().__init__(**kwargs)

        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def data_acquisition_iteration(self, model, Y: torch.Tensor, X):
        x_center = copy.deepcopy(X[Y.argmax(), :])
        weights = torch.ones_like(x_center)

        lb = self.task.lb * weights
        ub = self.task.ub * weights

        ei = qExpectedImprovement(model, Y.max().to(self.device))
        X_next, _ = optimize_acqf(ei,
                                  bounds=torch.stack([lb, ub]).to(self.device),
                                  q=self.batch_size,
                                  num_restarts=self.num_restarts,
                                  raw_samples=self.raw_samples)
        return X_next
