from typing import Tuple

import pandas as pd
import torch

from tasks.guacamol_objective_aabo import GuacamolObjective
# from tasks.hartmannn import Hartmann6D
from tasks.hartmannn_aabo import Hartmann6D
from tasks.lasso_dna_aabo import LassoDNA
from tasks.lunar_lander_aabo import LunarLander
from tasks.rover_aabo import RoverObjective
from trainers.base_trainer import BaseTrainer


class HartmannTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.task = Hartmann6D()
        self.num_initial_points = 100

    def initialize_data(self) -> Tuple[torch.tensor, torch.tensor]:
        init_train_x = torch.rand((self.num_initial_points,
                                   self.task.dim)).to(self.device) * (self.task.ub - self.task.lb) + self.task.lb
        init_train_y = self.task(init_train_x.to(self.device))

        return init_train_x, init_train_y

    def reinitialize_task(self):
        self.task = Hartmann6D()


class LunarTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.task = LunarLander()
        self.num_initial_points = 100

    def initialize_data(self) -> Tuple[torch.tensor, torch.tensor]:
        init_train_x = torch.rand((self.num_initial_points,
                                   self.task.dim)).to(self.device) * (self.task.ub - self.task.lb) + self.task.lb
        init_train_y = self.task(init_train_x.to(self.device))

        return init_train_x, init_train_y

    def reinitialize_task(self):
        self.task = LunarLander()


class RoverTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.task = RoverObjective()
        self.num_initial_points = 100

    def initialize_data(self) -> Tuple[torch.tensor, torch.tensor]:
        init_train_x = torch.rand((self.num_initial_points,
                                   self.task.dim)).to(self.device) * (self.task.ub - self.task.lb) + self.task.lb
        init_train_y = self.task(init_train_x.to(self.device))

        return init_train_x, init_train_y

    def reinitialize_task(self):
        self.task = RoverObjective()


class LassoDNATrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.task = LassoDNA()

    def initialize_data(self) -> Tuple[torch.tensor, torch.tensor]:
        init_train_x = torch.rand((self.num_initial_points,
                                   self.task.dim)).to(self.device) * (self.task.ub - self.task.lb) + self.task.lb
        init_train_y = self.task(init_train_x.to(self.device))

        return init_train_x, init_train_y

    def reinitialize_task(self):
        self.task = LassoDNA()


class GuacamolTrainer(BaseTrainer):

    def __init__(self, molecule, path_to_selfies_vae_files='src/tasks/utils/selfies_vae/', **kwargs):
        super().__init__(**kwargs)

        self.molecule = molecule
        self.path_to_selfies_vae_files = path_to_selfies_vae_files
        self.task = GuacamolObjective(guacamol_task_id=molecule,
                                      path_to_vae_statedict=self.path_to_selfies_vae_files +
                                      'selfies-vae-state-dict.pt',
                                      dtype=self.data_type)

    def initialize_data(self) -> Tuple[torch.tensor, torch.tensor]:
        # load guacamol data for initialization
        df = pd.read_csv(self.path_to_selfies_vae_files + "train_ys.csv")
        train_y = torch.from_numpy(df[self.molecule].values).to(dtype=self.data_type)
        train_x = torch.load(self.path_to_selfies_vae_files + "train_zs.pt").to(dtype=self.data_type)
        init_train_x = train_x[0:self.num_initial_points]
        init_train_y = train_y[0:self.num_initial_points]
        init_train_y, top_k_idxs = torch.topk(init_train_y, min(self.update_train_size, len(init_train_y)))
        init_train_x = init_train_x[top_k_idxs]
        init_train_y = init_train_y.unsqueeze(-1)
        self.task.num_calls = self.num_initial_points

        return init_train_x.to(self.device), init_train_y.to(self.device)

    def reinitialize_task(self):
        self.task = GuacamolObjective(guacamol_task_id=self.molecule,
                                      path_to_vae_statedict=self.path_to_selfies_vae_files +
                                      'selfies-vae-state-dict.pt',
                                      dtype=self.data_type)
