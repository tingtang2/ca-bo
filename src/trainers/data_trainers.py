from trainers.base_trainer import BaseTrainer
import torch
from typing import Tuple

from tasks.hartmannn import Hartmann6D


class HartmannTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.task = Hartmann6D(device=self.device)
        self.num_initial_points = 100

    def initialize_data(self) -> Tuple[torch.tensor, torch.tensor]:
        init_train_x = torch.rand((self.num_initial_points, self.task.dim)).to(
            self.device) * (self.task.upper_bound -
                            self.task.lower_bound) + self.task.lower_bound
        init_train_y = self.task.function_eval(init_train_x.to(self.device))

        # x dim and y dim need to be the same for botorch
        init_train_y = init_train_y.unsqueeze(1)
        return init_train_x, init_train_y
