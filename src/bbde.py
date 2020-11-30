from random import random
from math import exp

import numpy as np


def bbde(fobj, bounds, popsize=20, its=1000, alternative_exp_offset=True):
    exp_offset = 0.5 if alternative_exp_offset else 0.0
    results = []
    dimensions = len(bounds)

    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop = min_b + pop * diff                            # denormalisation

    fitness = np.asarray([fobj(ind) for ind in pop])
    best_index = np.argmin(fitness)
    best = pop[best_index]

    for i in range(its):
        for j in range(popsize):
            random_indexes = [idx for idx in range(popsize)]  # r1 != r2
            r1, r2 = pop[np.random.choice(random_indexes, 2, replace=False)]

            rand = random()                             # random real number between (0, 1)
            mutant = pop[j] + exp(rand - exp_offset) * (r1 - r2)     # (2) local selection

            trial = mutant                              # CR = 1 (1)

            f = fobj(trial)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_index]:
                    best_index = j
                    best = trial

        results.append((best, fitness[best_index]))

    return results
