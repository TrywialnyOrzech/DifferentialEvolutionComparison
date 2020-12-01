from progressbar import ProgressBar
from tqdm import tqdm
import numpy as np


class ClassicDE:

    def __init__(self, bounds, mutation, cross_probability, normalized_population, denorm_population, population_size, iterations, clip, exp_cross, cost_function):
        self.bounds = bounds
        self.mutation = mutation
        self.cross_probability = cross_probability,
        self.normalized_population = normalized_population
        self.denorm_population = denorm_population
        self.iterations = iterations
        self.clip = clip
        self.population_size = len(normalized_population)
        self.cost_function = cost_function
        self.dimensions = len(self.bounds)
        self.best_vector = np.empty([1])
        self.best_index = -1
        self.population_size = population_size
        self.do_exp = exp_cross
        self.best_vector = self.find_best_vector(self.denorm_population)

    def de(self):
        for i in tqdm(range(self.iterations)):
            for j in range(self.population_size):
                indexes_except_best = [ind for ind in range (self.population_size) if ind != j]
                vector_a, vector_b, vector_c = self.normalized_population[np.random.choice(indexes_except_best, 3, replace=False)]
                # Clip or Draw - to implement
                mutant = np.clip(vector_a + self.mutation * (vector_b - vector_c), 0, 1)
                if self.do_exp:
                    pass
                else:
                    cross_points = np.random.rand(self.dimensions) < self.cross_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimensions)] = True
                candidating_vect = np.where(cross_points, mutant, self.normalized_population[j])
                candidating_denorm = self.denorm(candidating_vect)
                self.update(self.cost_function(candidating_denorm), candidating_denorm, candidating_vect, j)
            yield self.best_vector, self.values_array[self.best_index]

    def denorm(self, normalized_population):
        # to implement: check if bounds are given in proper order
        min_bound, max_bound = np.asarray(self.bounds).T
        bounds_difference = np.fabs(min_bound - max_bound)
        return max_bound - bounds_difference * normalized_population

    def find_best_vector(self, denorm_population):
        self.values_array = np.asarray([self.cost_function(ind) for ind in denorm_population])
        self.best_index = np.argmin(self.values_array)
        return denorm_population[self.best_index]

    def update(self, candidate, denorm_candidate, norm_candidate, j):
        if candidate < self.values_array[j]:
            self.values_array[j] = candidate
            self.normalized_population[j] = norm_candidate
            if candidate < self.values_array[self.best_index]:
                self.best_index = j
                self.best_vector = denorm_candidate
