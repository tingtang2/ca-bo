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
                               likelihood=GaussianLikelihood().to(
                                   self.device)).to(self.device)

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
                f'Num oracle calls: {self.task.num_calls - 1}, best reward: {train_y.max().item():.3f}, final svgp loss: {final_loss:.3f}, epochs trained: {epochs_trained}, noise param: {self.model.likelihood.noise.item()}'
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


class SVGPEULBOTrainer(SVGPTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_inducing_points = 100
        self.grad_clip = 2.0

        self.early_stopping_threshold = 3
        self.train_batch_size = 32

        self.update_train_size = 100
        self.inducing_pt_init_w_moss23 = True

        self.alternative_updates = True

    def run_experiment(self, iteration: int):
        # get all attribute information
        logging.info(self.__dict__)
        train_x, train_y = self.initialize_data()

        # get inducing points
        inducing_points = train_x[:self.num_inducing_points]

        # init model
        self.model = SVGPModel(inducing_points=inducing_points,
                               likelihood=GaussianLikelihood().to(
                                   self.device)).to(self.device)
        if self.inducing_pt_init_w_moss23:
            optimal_inducing_points = self.get_optimal_inducing_points(
                model=self.model,
                prev_inducing_points=inducing_points,
            )
            self.model = SVGPModel(inducing_points=optimal_inducing_points,
                                   likelihood=GaussianLikelihood().to(
                                       self.device)).to(self.device)

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
            model_state_before_update = copy.deepcopy(model.state_dict())

            while (n_failures < 8) and (not success):
                try:
                    x_next = self.eulbo_train_model()
                    success = True
                except Exception as e:
                    # decrease lr to stabalize training
                    error_message = e
                    n_failures += 1
                    lr = lr / 10
                    x_next_lr = x_next_lr / 10
                    self.model.load_state_dict(
                        copy.deepcopy(model_state_before_update))
            if not success:
                assert 0, f"\nFailed to complete EULBO model update due to the following error:\n{error_message}"

            # Evaluate candidates
            y_next = self.task(x_next)

            # Update data
            train_x = torch.cat((train_x, x_next), dim=-2)
            train_y = torch.cat((train_y, y_next), dim=-2)

            logging.info(
                f'Num oracle calls: {self.task.num_calls - 1}, best reward: {train_y.max().item():.3f}, final svgp loss: {final_loss:.3f}, epochs trained: {epochs_trained}, noise param: {self.model.likelihood.noise.item()}'
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

    def eulbo_train_model(self, mll, loader):
        self.model.train()
        init_x_next = copy.deepcopy(init_x_next)
        x_next = Variable(init_x_next, requires_grad=True)

        base_samples = torch.randn(num_mc_samples_qei,
                                   acquisition_bsz).to(device=self.device)

        if ablation1_fix_indpts_and_hypers:
            model_params_to_update = list(model.variational_parameters())
        elif ablation2_fix_hypers:
            model_params_to_update = list(model.variational_parameters()) + [
                model.variational_strategy.inducing_points
            ]
        else:
            model_params_to_update = list(model.parameters())
        lowest_loss = torch.inf
        n_failures_improve_loss = 0
        epochs_trained = 0
        continue_training_condition = True
        if (max_allowed_n_epochs == 0) or (n_epochs == 0):
            continue_training_condition = False
        currently_training_model = True
        x_next_optimizer = torch.optim.Adam([
            {
                'params': x_next
            },
        ],
                                            lr=x_next_lr)
        model_optimizer = torch.optim.Adam([{
            'params': model_params_to_update,
        }],
                                           lr=lr)
        joint_optimizer = torch.optim.Adam([{
            'params': x_next,
        }, {
            'params': model_params_to_update,
        }],
                                           lr=lr)
        kg_samples = None
        zs = None
        while continue_training_condition:
            total_loss = 0
            for (inputs, scores) in loader:
                if self.alternate_updates:
                    model_optimizer.zero_grad()
                    x_next_optimizer.zero_grad()
                else:
                    joint_optimizer.zero_grad()
                output = self.model(inputs.to(self.device))
                nelbo = -mll(output, scores.to(self.device))

                expected_log_utility_x_next = get_expected_log_utility_ei()
                loss = nelbo - expected_log_utility_x_next
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(x_next,
                                                   max_norm=self.grad_clip)
                if self.alternate_updates:
                    if currently_training_model:
                        model_optimizer.step()
                    else:
                        x_next_optimizer.step()
                else:
                    joint_optimizer.step()
                with torch.no_grad():
                    x_next[:, :] = x_next.clamp(self.task.lb, self.task.ub)
                    total_loss += loss.item()
            epochs_trained += 1
            currently_training_model = not currently_training_model
            if total_loss < lowest_loss:
                lowest_loss = total_loss
            else:
                n_failures_improve_loss += 1
            continue_training_condition = n_failures_improve_loss < max_allowed_n_failures_improve_loss
            if epochs_trained > max_allowed_n_epochs:
                continue_training_condition = False

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
