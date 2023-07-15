import warnings
import autograd
import autograd.numpy as anp
import tensorflow as tf

class Problem:

    def __init__(self, n_var=-1, n_obj=-1, n_constr=0, xl=None, xu=None, type_var=tf.float32):
        self.n_var = n_var
        self.type_var = type_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self._pareto_front = None
        self._pareto_set = None
        if xl is None:
            self.xl = tf.zeros(n_var, dtype=type_var)
        else:
            self.xl = tf.constant(xl, dtype=type_var)
        if xu is None:
            self.xu = tf.ones(n_var, dtype=type_var)
        else:
            self.xu = tf.constant(xu, dtype=type_var)

        self.evaluation_of = ["F"]
        if self.n_constr > 0:
            self.evaluation_of.append("G")

    def pareto_front(self, *args, **kwargs):
        if self._pareto_front is None:
            self._pareto_front = self._calc_pareto_front(*args, **kwargs)

        return self._pareto_front

    def pareto_set(self, *args, **kwargs):
        if self._pareto_set is None:
            self._pareto_set = self._calc_pareto_set(*args, **kwargs)

        return self._pareto_set

    def evaluate(self,
                 X,
                 *args,
                 return_values_of="auto",
                 return_as_dictionary=False,
                 **kwargs):

        # make the array at least 2-d - even if only one row should be evaluated
        only_single_value = len(tf.shape(X)) == 1
        X = tf.expand_dims(X, axis=0)

        # check the dimensionality of the problem and the given input
        if tf.shape(X)[1] != self.n_var:
            raise Exception('Input dimension %s is not equal to n_var %s!' % (tf.shape(X)[1], self.n_var))

        # automatic return the function values and CV if it has constraints if not defined otherwise
        if type(return_values_of) == str and return_values_of == "auto":
            return_values_of = ["F"]
            if self.n_constr > 0:
                return_values_of.append("CV")

        # create the output dictionary for _evaluate to be filled
        out = {}
        for val in return_values_of:
            out[val] = None

        # all values that are set in the evaluation function
        values_not_set = [val for val in return_values_of if val not in self.evaluation_of]

        self._evaluate(X, out, *args, **kwargs)
        self.at_least2d(out)

        # if constraint violation should be returned as well
        if self.n_constr == 0:
            CV = tf.zeros([tf.shape(X)[0], 1])
        else:
            CV = Problem.calc_constraint_violation(out["G"])

        if "CV" in return_values_of:
            out["CV"] = CV

        # if an additional boolean flag for feasibility should be returned
        if "feasible" in return_values_of:
            out["feasible"] = (CV <= 0)

        # remove the first dimension of the output - in case input was a 1d- vector
        if only_single_value:
            for key in out.keys():
                if out[key] is not None:
                    out[key] = out[key][0, :]

        if return_as_dictionary:
            return out
        else:
            # if just a single value, do not return a tuple
            if len(return_values_of) == 1:
                return out[return_values_of[0]]
            else:
                return tuple([out[val] for val in return_values_of])

    @staticmethod
    def at_least2d(d):
        for key in d.keys():
            if len(tf.shape(d[key])) == 1:
                d[key] = tf.expand_dims(d[key], axis=0)

    @staticmethod
    def calc_constraint_violation(G):
        if G is None:
            return None
        elif tf.shape(G)[1] == 0:
            return tf.zeros(tf.shape(G)[0])[:, None]
        else:
            return tf.reduce_sum(G * tf.cast(G > 0, dtype=G.dtype), axis=1)[:, None]

# Make sure the output is at least 2D
def at_least2d(d):
    for key in d.keys():
        if len(tf.shape(d[key])) == 1:
            d[key] = tf.expand_dims(d[key], axis=0)

