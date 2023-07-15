import random
import numpy as np
import pandas as pd
import tensorflow as tf
from deap import base, creator, algorithms, tools
from pymop.factory import get_problem

PROBLEM = "c1dtlz1"
NOBJ = 3
K = 10
NDIM = NOBJ + K - 1
P = 12

def factorial(x):
    return np.math.factorial(x)

H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 0.0, 1.0
problem = get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)

MU = int(H + (4 - H % 4))
NGEN = 400
CXPB = 1.0
MUTPB = 1.0

ref_points = tools.uniform_reference_points(NOBJ, P)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)

def uniform(low, up, size=None):
    try:
        size = tuple(int(s) for s in size)  # Convert size to integers
    except TypeError:
        pass
    return tf.random.uniform(shape=size, minval=low, maxval=up)


toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, size=NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

def main(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df = pd.DataFrame()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)
    pop = tf.convert_to_tensor(pop)  # Convert to TensorFlow tensor

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, MU)
        pop = tf.convert_to_tensor(pop)  # Convert to TensorFlow tensor

        F = problem.evaluate(pop, return_values_of=["F"])
        new = pd.DataFrame({'gen': [gen], 'mean': F.mean(), 'min': F.min()})
        df = pd.concat([df, new])

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    df.to_csv('df.csv')
    return pop, logbook

with tf.device('/GPU:0'):
    pop, logbook = main(seed=1)
