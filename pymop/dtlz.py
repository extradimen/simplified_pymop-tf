import tensorflow as tf

from pymop.problem import Problem


class DTLZ(Problem):
    def __init__(self, n_var, n_obj, k=None):

        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=tf.float32)

    def g1(self, X_M):
        return 100 * (self.k + tf.reduce_sum(tf.square(X_M - 0.5) - tf.cos(20 * tf.constant(3.141592653589793, dtype=tf.float32) * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return tf.reduce_sum(tf.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= tf.reduce_prod(tf.cos(tf.pow(X_[:, :X_.shape[1] - i], alpha) * tf.constant(3.141592653589793, dtype=tf.float32) / 2.0), axis=1)
            if i > 0:
                _f *= tf.sin(tf.pow(X_[:, X_.shape[1] - i], alpha) * tf.constant(3.141592653589793, dtype=tf.float32) / 2.0)

            f.append(_f)

        f = tf.stack(f, axis=1)
        return f


class DTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        return 0.5 * ref_dirs

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)

        f = []
        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= tf.reduce_prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        out["F"] = tf.stack(f, axis=1)
