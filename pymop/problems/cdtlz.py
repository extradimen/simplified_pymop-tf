import tensorflow as tf

from pymop.problems.dtlz import DTLZ1


class C1DTLZ1(DTLZ1):

    def __init__(self, n_var=12, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.n_constr = 1

    def _evaluate(self, X, out, *args, **kwargs):
        super()._evaluate(X, out, *args, **kwargs)
        out["G"] = constraint_c1_linear(out["F"])

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        return super()._calc_pareto_front(ref_dirs, *args, **kwargs)

def constraint_c1_linear(f):
    g = - (1 - f[:, -1] / 0.6 - tf.reduce_sum(f[:, :-1] / 0.5, axis=1))

    return g
