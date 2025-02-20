import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Union

import torch

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
                 norm_data: bool = False,
                 tracker=None,
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
        self.norm_data = norm_data

        # wandb tracking
        self.tracker = tracker

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
    def run_experiment(self, iteration: int):
        pass

    @abstractmethod
    def initialize_data(self):
        pass

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), f'{self.save_dir}models/{name}.pt')

    def save_metrics(self, metrics: List[float], iter: int, name: str):
        save_name = f'{name}_iteration_{iter}-{datetime.now().strftime("%m_%d_%Y_%H:%M:%S")}.json'
        with open(Path(Path.home(), self.save_dir, 'metrics/', save_name),
                  'w') as f:
            json.dump(metrics, f)

    def init_new_run(self, tracker):
        self.tracker = tracker

    def log_wandb_metrics(self,
                          train_y: torch.Tensor,
                          final_loss: float = -1,
                          epochs_trained: int = -1,
                          train_rmse: float = -1.0,
                          log_to_file: bool = True,
                          model=None):

        if 'exact' in self.trainer_type:
            passed_model = model
        else:
            passed_model = self.model

        raw_outputscale = passed_model.covar_module.raw_outputscale
        constraint = passed_model.covar_module.raw_outputscale_constraint
        outputscale = constraint.transform(raw_outputscale)

        raw_lengthscale = passed_model.covar_module.base_kernel.raw_lengthscale
        constraint = passed_model.covar_module.base_kernel.raw_lengthscale_constraint
        lengthscale = constraint.transform(raw_lengthscale)

        if 'exact' in self.trainer_type:
            log_dict = {
                'Num oracle calls': self.task.num_calls - 1,
                'best reward': train_y.max().item(),
                'noise param': passed_model.likelihood.noise.item(),
                'lengthscale param': lengthscale.item(),
                'outputscale param': outputscale.item(),
                'train rmse': train_rmse
            }
        else:
            log_dict = {
                'Num oracle calls': self.task.num_calls - 1,
                'best reward': train_y.max().item(),
                'final svgp loss': final_loss,
                'epochs trained': epochs_trained,
                'noise param': self.model.likelihood.noise.item(),
                'lengthscale param': lengthscale.item(),
                'outputscale param': outputscale.item(),
                'train rmse': train_rmse
            }

        if not self.turn_off_wandb:
            self.tracker.log(log_dict)

        if log_to_file:
            logging.info(', '.join(
                [f'{key}: {value:.5f}' for key, value in log_dict.items()]))
