from trainers.base_trainer import BaseTrainer


class ExactGPTrainer(BaseTrainer):

    def run_experiment(self):
        return super().run_experiment()

    def data_acquisition_iteration(self):
        return super().data_acquisition_iteration()

    def eval(self):
        return super().eval()
