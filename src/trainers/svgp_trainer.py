import copy
import logging

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from models.svgp import SVGPModel
from trainers.acquisition_fn_trainers import EITrainer
from trainers.base_trainer import BaseTrainer
from trainers.data_trainers import HartmannTrainer
from trainers.utils.expected_log_utility import get_expected_log_utility_ei
from trainers.utils.moss_et_al_inducing_pts_init import \
    GreedyImprovementReduction


class SVGPRetrainTrainer(BaseTrainer):

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

        reward = []
        for i in trange(self.max_oracle_calls - self.num_initial_points):
            if self.norm_data:
                # get normalized train y
                train_y_mean = train_y.mean()
                train_y_std = train_y.std()
                if train_y_std == 0:
                    train_y_std = 1
                train_y = (train_y - train_y_mean) / train_y_std
            # reinit model
            self.model = SVGPModel(
                inducing_points=train_x[-(train_x.size(0) // 2):],
                likelihood=GaussianLikelihood().to(self.device),
                kernel_type=self.kernel_type).to(self.device)

            self.optimizer = self.optimizer_type(
                [{
                    'params': self.model.parameters(),
                }], lr=self.learning_rate)

            mll = VariationalELBO(self.model.likelihood,
                                  self.model,
                                  num_data=train_x.size(0))

            train_loader = self.generate_dataloaders(train_x=train_x,
                                                     train_y=train_y.squeeze())

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            x_next = self.data_acquisition_iteration(self.model, train_y,
                                                     train_x)

            # Evaluate candidates
            y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            logging.info(
                f'Num oracle calls: {self.task.num_calls - 1}, best reward: {train_y.max().item():.3f}, final svgp loss: {final_loss:.3f}, epochs trained: {epochs_trained}, noise param: {self.model.likelihood.noise.item():.6f}'
            )

            if not self.turn_off_wandb:
                self.tracker.log({
                    'Num oracle calls':
                    self.task.num_calls - 1,
                    'best reward':
                    train_y.max().item(),
                    'final svgp loss':
                    final_loss,
                    'epochs trained':
                    epochs_trained,
                    'noise param':
                    self.model.likelihood.noise.item()
                })
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

    def get_optimal_inducing_points(self, prev_inducing_points):
        greedy_imp_reduction = GreedyImprovementReduction(
            model=self.model,
            maximize=True,
        )
        optimal_inducing_points = greedy_imp_reduction.allocate_inducing_points(
            inputs=prev_inducing_points.to(self.device),
            covar_module=self.model.covar_module,
            num_inducing=prev_inducing_points.shape[0],
            input_batch_shape=1,
        )
        return optimal_inducing_points


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
        inducing_points = train_x[:self.num_inducing_points]

        # init model
        self.model = SVGPModel(inducing_points=inducing_points,
                               likelihood=GaussianLikelihood().to(self.device),
                               kernel_type=self.kernel_type).to(self.device)

        self.optimizer = self.optimizer_type(
            [{
                'params': self.model.parameters(),
                # 'params': self.model.likelihood.parameters()
            }],
            lr=self.learning_rate)

        reward = []
        for i in trange(self.max_oracle_calls - self.num_initial_points):
            if self.norm_data:
                # get normalized train y
                train_y_mean = train_y.mean()
                train_y_std = train_y.std()
                if train_y_std == 0:
                    train_y_std = 1
                train_y = (train_y - train_y_mean) / train_y_std

            # only update on recently acquired points
            if i > 0:
                update_x = train_x[-self.update_train_size:]
                # y needs to only have 1 dimension when training in gpytorch
                update_y = train_y.squeeze()[-self.update_train_size:]
            else:
                update_x = train_x
                update_y = train_y.squeeze()

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

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            logging.info(
                f'Num oracle calls: {self.task.num_calls - 1}, best reward: {train_y.max().item():.3f}, final svgp loss: {final_loss:.3f}, epochs trained: {epochs_trained}, noise param: {self.model.likelihood.noise.item():.6f}'
            )

            if not self.turn_off_wandb:
                self.tracker.log({
                    'Num oracle calls':
                    self.task.num_calls - 1,
                    'best reward':
                    train_y.max().item(),
                    'final svgp loss':
                    final_loss,
                    'epochs trained':
                    epochs_trained,
                    'noise param':
                    self.model.likelihood.noise.item()
                })
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

    def get_optimal_inducing_points(self, prev_inducing_points):
        greedy_imp_reduction = GreedyImprovementReduction(
            model=self.model,
            maximize=True,
        )
        optimal_inducing_points = greedy_imp_reduction.allocate_inducing_points(
            inputs=prev_inducing_points.to(self.device),
            covar_module=self.model.covar_module,
            num_inducing=prev_inducing_points.shape[0],
            input_batch_shape=1,
        )
        return optimal_inducing_points


class HartmannEISVGPTrainer(SVGPTrainer, HartmannTrainer, EITrainer):
    pass


class HartmannEISVGPRetrainTrainer(SVGPRetrainTrainer, HartmannTrainer,
                                   EITrainer):
    pass


class SVGPEULBOTrainer(SVGPTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_inducing_points = 100
        self.grad_clip = 2.0

        self.early_stopping_threshold = 3
        self.train_batch_size = 32

        self.update_train_size = 100
        self.inducing_pt_init_w_moss23 = True

        self.alternate_updates = True
        self.eulbo_epochs = 30
        self.x_next_lr = 0.001

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()

        # get inducing points
        inducing_points = train_x[:self.num_inducing_points]

        # init model
        self.model = SVGPModel(inducing_points=inducing_points,
                               likelihood=GaussianLikelihood().to(self.device),
                               kernel_type=self.kernel_type).to(self.device)
        if self.inducing_pt_init_w_moss23:
            optimal_inducing_points = self.get_optimal_inducing_points(
                prev_inducing_points=inducing_points)
            self.model = SVGPModel(
                inducing_points=optimal_inducing_points,
                likelihood=GaussianLikelihood().to(self.device),
                kernel_type=self.kernel_type).to(self.device)

        self.optimizer = self.optimizer_type(
            [{
                'params': self.model.parameters(),
            }], lr=self.learning_rate)

        reward = []
        for i in trange(self.max_oracle_calls - self.num_initial_points):
            if self.norm_data:
                # get normalized train y
                train_y_mean = train_y.mean()
                train_y_std = train_y.std()
                if train_y_std == 0:
                    train_y_std = 1
                train_y = (train_y - train_y_mean) / train_y_std

            # only update on recently acquired points
            if i > 0:
                update_x = train_x[-self.update_train_size:]
                # y needs to only have 1 dimension when training in gpytorch
                update_y = train_y.squeeze()[-self.update_train_size:]
            else:
                update_x = train_x
                update_y = train_y.squeeze()

            mll = VariationalELBO(self.model.likelihood,
                                  self.model,
                                  num_data=update_x.size(0))

            train_loader = self.generate_dataloaders(train_x=update_x,
                                                     train_y=update_y)

            final_loss, epochs_trained = self.train_model(train_loader, mll)
            self.model.eval()

            x_next = self.data_acquisition_iteration(self.model, train_y,
                                                     train_x)

            # above is warm start
            torch.autograd.set_detect_anomaly(True)

            n_failures = 0
            success = False
            model_state_before_update = copy.deepcopy(self.model.state_dict())

            while (n_failures < 8) and (not success):
                try:
                    x_next, final_loss, epochs_trained = self.eulbo_train_model(
                        mll=mll,
                        loader=train_loader,
                        train_y=train_y,
                        init_x_next=x_next)
                    success = True
                except Exception as e:
                    # decrease lr to stabalize training
                    error_message = e
                    logging.info(f'adjusting lrs: {e}')
                    n_failures += 1
                    self.learning_rate = self.learning_rate / 10
                    self.x_next_lr = self.x_next_lr / 10
                    self.model.load_state_dict(
                        copy.deepcopy(model_state_before_update))
            if not success:
                assert 0, f"\nFailed to complete EULBO model update due to the following error:\n{error_message}"
            self.model.eval()

            # Evaluate candidates
            y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            logging.info(
                f'Num oracle calls: {self.task.num_calls - 1}, best reward: {train_y.max().item():.3f}, final svgp loss: {final_loss:.3f}, epochs trained: {epochs_trained}, noise param: {self.model.likelihood.noise.item():.6f}'
            )

            if not self.turn_off_wandb:
                self.tracker.log({
                    'Num oracle calls':
                    self.task.num_calls - 1,
                    'best reward':
                    train_y.max().item(),
                    'final svgp loss':
                    final_loss,
                    'epochs trained':
                    epochs_trained,
                    'noise param':
                    self.model.likelihood.noise.item()
                })
            reward.append(train_y.max().item())

        self.save_metrics(metrics=reward,
                          iter=iteration,
                          name=self.trainer_type)

    def eulbo_train_model(self, loader, mll, train_y, init_x_next):
        self.model.train()
        init_x_next = copy.deepcopy(init_x_next)
        x_next = Variable(init_x_next, requires_grad=True)

        model_params_to_update = list(self.model.parameters())
        self.x_next_optimizer = torch.optim.Adam([
            {
                'params': x_next
            },
        ],
                                                 lr=self.x_next_lr)
        self.model_optimizer = torch.optim.Adam(
            [{
                'params': model_params_to_update,
            }], lr=self.learning_rate)
        self.joint_optimizer = torch.optim.Adam(
            [{
                'params': x_next,
            }, {
                'params': model_params_to_update,
            }],
            lr=self.learning_rate)

        best_loss = 1e+5
        early_stopping_counter = 0
        currently_training_model = True
        for i in range(self.eulbo_epochs):
            loss = self.eulbo_train_epoch(
                loader,
                mll,
                x_next,
                currently_training_model=currently_training_model,
                train_y=train_y)

            currently_training_model = not currently_training_model
            if loss < best_loss:
                # self.save_model(f'{self.name}_{iter}')
                early_stopping_counter = 0
                best_loss = loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                return x_next.detach(), loss, i + 1

        return x_next.detach(), loss, i + 1

    def eulbo_train_epoch(self, loader, mll, x_next, currently_training_model,
                          train_y):
        total_loss = 0
        for i, (inputs, scores) in enumerate(loader):
            if self.alternate_updates:
                self.model_optimizer.zero_grad()
                self.x_next_optimizer.zero_grad()
            else:
                self.joint_optimizer.zero_grad()
            output = self.model(inputs.to(self.device))
            nelbo = -mll(output, scores.to(self.device))

            expected_log_utility_x_next = get_expected_log_utility_ei(
                self.model,
                best_f=train_y.max(),
                x_next=x_next,
                device=self.device)
            loss = nelbo - expected_log_utility_x_next
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.grad_clip)
                torch.nn.utils.clip_grad_norm_(x_next, max_norm=self.grad_clip)
            if self.alternate_updates:
                if currently_training_model:
                    self.model_optimizer.step()
                else:
                    self.x_next_optimizer.step()
            else:
                self.joint_optimizer.step()

            with torch.no_grad():
                x_next[:, :] = x_next.clamp(self.task.lb, self.task.ub)
                total_loss += loss.item()

        return total_loss

    def train_model(self, train_loader: DataLoader, mll):
        self.model.train()
        for i in range(self.epochs):
            loss = self.train_epoch(train_loader, mll)
        return loss, i + 1

    def train_epoch(self, train_loader: DataLoader, mll):
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            self.optimizer.zero_grad()

            output = self.model(x.to(self.device))
            loss = -mll(output, y.to(self.device))

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


class HartmannEISVGPEULBOTrainer(SVGPEULBOTrainer, HartmannTrainer, EITrainer):
    pass
