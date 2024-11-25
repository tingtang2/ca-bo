from typing import Tuple

import torch

# from tasks.hartmannn import Hartmann6D
from tasks.hartmannn_aabo import Hartmann6D
from trainers.base_trainer import BaseTrainer


class HartmannTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.task = Hartmann6D()
        self.num_initial_points = 100

    def initialize_data(self) -> Tuple[torch.tensor, torch.tensor]:
        init_train_x = torch.rand((self.num_initial_points, self.task.dim)).to(
            self.device) * (self.task.ub - self.task.lb) + self.task.lb
        init_train_y = self.task(init_train_x.to(self.device))

        return init_train_x, init_train_y
