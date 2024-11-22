from botorch.test_functions import Hartmann

from tasks.task import Task


class Hartmann6D(Task):

    def __init__(self):
        self.neg_hartmann6 = Hartmann(negate=True)

        super().__init__(dim=6, upper_bound=1, lower_bound=0)

    def function_eval(self, x):
        # known optimum at [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
        self.num_calls += 1
        y = self.neg_hartmann6(x)
        return y.item()
