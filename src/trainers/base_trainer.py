from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

from tasks.hartmannn import Hartmann6D
from tasks.task import Task


class BaseTrainer(ABC):

    def __init__(self,
                 optimizer_type,
                 criterion,
                 device: str,
                 save_dir: Union[str, Path],
                 batch_size: int,
                 dropout_prob: float,
                 learning_rate: float,
                 max_oracle_calls: int,
                 save_plots: bool = True,
                 seed: int = 11202022,
                 **kwargs) -> None:
        super().__init__()

        # basic configs every trainer needs
        self.optimizer_type = optimizer_type
        self.criterion = criterion
        self.device = torch.device(device)
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.seed = seed
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate

        # BO specific
        self.task: Task = None
        self.max_oracle_calls = max_oracle_calls

        # extra configs in form of kwargs
        for key, item in kwargs.items():
            setattr(self, key, item)

    @abstractmethod
    def data_acquisition_iteration(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def run_experiment(self):
        pass

    @abstractmethod
    def initialize_data(self):
        pass

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), f'{self.save_dir}models/{name}.pt')


class HartmannTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.task = Hartmann6D()
        self.num_initial_points = 100

    def initialize_data(self) -> Tuple[torch.tensor, torch.tensor]:
        init_train_x = torch.rand(self.num_initial_points, self.task.dim) * (
            self.task.upper_bound -
            self.task.lower_bound) + self.task.lower_bound
        init_train_y = self.task.function_eval(init_train_x.to(self.device))

        return init_train_x, init_train_y


class EITrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_restarts = 10
        self.raw_samples = 256

    def data_acquisition_iteration(self, model, Y: torch.Tensor):
        ei = qExpectedImprovement(model, Y.max().to(self.device))
        X_next, _ = optimize_acqf(
            ei,
            bounds=torch.stack([self.task.lower_bound,
                                self.task.upper_bound]).to(self.device),
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return X_next
