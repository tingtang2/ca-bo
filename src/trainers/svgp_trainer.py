import logging

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from models.svgp import SVGPModel
from trainers.acquisition_fn_trainers import EITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import HartmannTrainer


class SVGPTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_inducing_points = 100
        self.grad_clip = 2.0

        self.early_stopping_threshold = 3
        self.train_batch_size = 32

        self.update_train_size = 100

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()

        # get inducing points
        inducing_points = train_x[0:self.num_inducing_points, :]

        # init model
        self.model = SVGPModel(inducing_points=inducing_points,
                               likelihood=GaussianLikelihood().to(
                                   self.device)).to(self.device)
        self.optimizer = self.optimizer_type(
            [{
                'params': self.model.parameters(),
                # 'params': self.model.likelihood.parameters()
            }],
            lr=self.learning_rate)
        self.init_training_complete = False

        reward = []
        for i in trange(self.max_oracle_calls):
            if self.norm_data:
                # get normalized train y
                train_y_mean = train_y.mean()
                train_y_std = train_y.std()
                if train_y_std == 0:
                    train_y_std = 1
                train_y = (train_y - train_y_mean) / train_y_std

            # only update on recently acquired points
            if not self.init_training_complete:
                update_x = train_x
                update_y = train_y.squeeze()
                self.init_training_complete = True
            else:
                update_x = train_x[-self.update_train_size:]
                update_y = train_y.squeeze()[-self.update_train_size:]

            # if i > 0:
            #     update_x = train_x[-self.update_train_size:]
            #     update_y = train_y[-self.update_train_size:]
            # else:
            #     update_x = train_x
            #     update_y = train_y

            mll = VariationalELBO(self.model.likelihood,
                                  self.model,
                                  num_data=update_x.size(0))

            train_loader = self.generate_dataloaders(train_x=update_x,
                                                     train_y=update_y)

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            x_next = self.data_acquisition_iteration(self.model, train_y,
                                                     train_x)

            # Evaluate candidates
            y_next = self.task(x_next)
            # y_next = y_next.unsqueeze(-1)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            logging.info(
                f'Num oracle calls: {self.task.num_calls - 1}, best reward: {train_y.max().item():.3f}, final svgp loss: {final_loss:.3f}, epochs trained: {epochs_trained}'
            )
            reward.append(train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    def train_model(self, train_loader: DataLoader, mll):
        self.model.train()
        best_loss = 1e+5
        early_stopping_counter = 0
        for i in range(self.epochs):
            loss = self.train_epoch(train_loader, mll)
            # logging.info(f'epoch: {i} training loss: {loss:.3f}')

            if loss < best_loss:
                # self.save_model(f'{self.name}_{iter}')
                early_stopping_counter = 0
                best_loss = loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                return loss, i + 1

        return loss, i + 1

    def train_epoch(self, train_loader: DataLoader, mll):
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            self.optimizer.zero_grad()

            output = self.model(x.to(self.device))
            loss = -mll(output, y.to(self.device))
            # loss = loss.sum()

            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.grad_clip)

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss

    def generate_dataloaders(self, train_x, train_y):
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_loader

    def eval(self):
        pass


class HartmannEISVGPTrainer(SVGPTrainer, HartmannTrainer, EITrainer):
    pass
