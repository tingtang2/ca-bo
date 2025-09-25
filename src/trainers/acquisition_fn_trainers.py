import copy

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.optim.initializers import initialize_q_batch_nonneg

from set_seed import set_seed
from trainers.base_trainer import BaseTrainer
from trainers.utils.analytic_log_ei import LogExpectedImprovement
from trainers.utils.stochastic_sampler import StochasticSampler


class EITrainer(BaseTrainer):

    def data_acquisition_iteration(self,
                                   model,
                                   Y: torch.Tensor,
                                   X,
                                   num_restarts: int = 10,
                                   raw_samples: int = 256):
        x_center = copy.deepcopy(X[Y.argmax(), :])
        weights = torch.ones_like(x_center)

        lb = self.task.lb * weights
        ub = self.task.ub * weights

        if self.use_analytic_acq_func:
            ei = ExpectedImprovement(model, Y.max().to(self.device))
        else:
            ei = qExpectedImprovement(model, Y.max().to(self.device))

        if self.enable_raasp:
            options = {'sample_around_best': True}
        else:
            options = None

        X_next, acq_val, origin = optimize_acqf(ei,
                                                bounds=torch.stack(
                                                    [lb, ub]).to(self.device),
                                                q=self.batch_size,
                                                num_restarts=num_restarts,
                                                raw_samples=raw_samples,
                                                options=options)
        return X_next.detach(), acq_val.detach(), origin


class LogEITrainer(BaseTrainer):

    def data_acquisition_iteration(self,
                                   model,
                                   Y: torch.Tensor,
                                   X,
                                   num_restarts: int = 10,
                                   raw_samples: int = 256):
        x_center = copy.deepcopy(X[Y.argmax(), :])
        weights = torch.ones_like(x_center)

        lb = self.task.lb * weights
        ub = self.task.ub * weights

        if self.use_analytic_acq_func:
            ei = LogExpectedImprovement(model, Y.max().to(self.device))
        else:
            ei = qLogExpectedImprovement(model, Y.max().to(self.device))

        if self.enable_raasp:
            options = {'sample_around_best': True}
        else:
            options = None

        X_next, acq_val, origin = optimize_acqf(
            ei,
            bounds=torch.stack([lb, ub]).to(self.device, self.data_type),
            q=self.batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options)
        return X_next.detach(), acq_val.detach(), origin


class CustomEITrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data_acquisition_iteration(self,
                                   model,
                                   Y: torch.Tensor,
                                   X,
                                   num_restarts: int = 10,
                                   raw_samples: int = 256):
        x_center = copy.deepcopy(X[Y.argmax(), :])
        weights = torch.ones_like(x_center)

        lb = self.task.lb * weights
        ub = self.task.ub * weights

        sampler = StochasticSampler(sample_shape=torch.Size([1]))
        qEI = qExpectedImprovement(model,
                                   best_f=Y.max().to(self.device),
                                   sampler=sampler)

        # generate a large number of random q-batches
        Xraw = lb + (ub - lb) * torch.rand(num_restarts * raw_samples,
                                           self.batch_size, self.task.dim).to(
                                               self.device)
        Yraw = qEI(
            Xraw)  # evaluate the acquisition function on these q-batches

        # apply the heuristic for sampling promising initial conditions
        X_new = initialize_q_batch_nonneg(Xraw, Yraw, num_restarts)
        # X_new = X_new.squeeze(1)

        # we'll want gradients for the input
        X_new.requires_grad_(True)

        # set up the optimizer, make sure to only pass in the candidate set here
        optimizer = torch.optim.Adam([X_new], lr=0.01)
        X_traj = []  # we'll store the results

        # run a basic optimization loop
        for i in range(75):
            optimizer.zero_grad()
            # this performs batch evaluation, so this is an N-dim tensor
            print(X_new.shape)
            losses = -qEI(X_new)  # torch.optim minimizes
            loss = losses.sum()

            loss.backward()  # perform backward pass
            optimizer.step()  # take a step

            # clamp values to the feasible set
            for j, (l, u) in enumerate((lb, ub)):
                X_new.data[..., j].clamp_(
                    l, u)  # need to do this on the data not X itself

            # store the optimization trajecatory
            X_traj.append(X_new.detach().clone())

            if (i + 1) % 15 == 0:
                print(f"Iteration {i+1:>3}/75 - Loss: {loss.item():>4.3f}")
        self.save_metrics(X_traj, 0, name='x_trajectory')

        return X_new
