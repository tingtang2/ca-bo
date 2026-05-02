import torch

from trainers.utils.turbo import TurboState, get_trust_region_bounds


class TurboTrainerMixin:

    def initialize_turbo_state(self, train_x, train_y):
        self.tr_state = TurboState(dim=train_x.shape[-1],
                                   batch_size=self.batch_size,
                                   best_value=train_y.max().item())

    def set_local_train_y_stats(self, train_y):
        self.train_y_mean = train_y.mean()
        self.train_y_std = train_y.std()
        if self.train_y_std == 0:
            self.train_y_std = 1

    def restart_local_trust_region(self):
        if self.turn_on_sobol_init:
            local_train_x, local_train_y = self.sobol_initialize_data()
        else:
            local_train_x = torch.rand((self.num_initial_points, self.task.dim)
                                       ).to(self.device, self.data_type)
            if self.turn_on_simple_input_transform:
                eval_x = local_train_x * (self.task.ub - self.task.lb
                                          ) + self.task.lb
            else:
                local_train_x = local_train_x * (self.task.ub - self.task.lb
                                                 ) + self.task.lb
                eval_x = local_train_x
            local_train_y = self.task(eval_x)

        self.set_local_train_y_stats(local_train_y)
        self.initialize_turbo_state(train_x=local_train_x, train_y=local_train_y)
        return local_train_x, local_train_y

    def get_acq_bounds(self, model, Y: torch.Tensor, X):
        if self.turn_on_simple_input_transform:
            lb = torch.zeros(X.shape[-1], device=X.device, dtype=X.dtype)
            ub = torch.ones(X.shape[-1], device=X.device, dtype=X.dtype)
        else:
            lb = self.task.lb.to(X.device, X.dtype)
            ub = self.task.ub.to(X.device, X.dtype)

        return get_trust_region_bounds(model=model,
                                       X=X,
                                       Y=Y,
                                       length=self.tr_state.length,
                                       lb=lb,
                                       ub=ub)
