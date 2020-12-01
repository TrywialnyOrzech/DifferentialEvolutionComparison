from random import random
from math import exp
from tqdm import tqdm

import numpy as np


def bbde(fobj, bounds, population, its=1000, alternative_exp_offset=True):
    exp_offset = 0.5 if alternative_exp_offset else 0.0
    results = []
    popsize = len(population)

    fitness = np.asarray([fobj(ind) for ind in population])
    best_index = np.argmin(fitness)
    best = population[best_index]

    for i in tqdm(range(its), leave=False):
        for j in range(len(population)):
            random_indexes = [idx for idx in range(popsize)]  # r1 != r2
            r1, r2 = population[np.random.choice(random_indexes, 2, replace=False)]

            rand = random()                             # random real number between (0, 1)
            mutant = population[j] + exp(rand - exp_offset) * (r1 - r2)     # (2) local selection

            trial = mutant                              # CR = 1 (1)

            f = fobj(trial)
            if f < fitness[j]:
                fitness[j] = f
                population[j] = trial
                if f < fitness[best_index]:
                    best_index = j
                    best = trial

        results.append((best, fitness[best_index]))

    return results
