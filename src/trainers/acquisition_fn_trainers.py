import torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

from trainers.base_trainer import BaseTrainer
import copy


class EITrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_restarts = 10
        self.raw_samples = 256

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
