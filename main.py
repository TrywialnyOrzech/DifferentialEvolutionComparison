import matplotlib.pyplot as plt
import numpy as np
from src.de import ClassicDE


def main():
    bounds = [(-10, 10)] * 10
    mutation = 0.8
    cross_probability = 0.7
    population_size = 20
    iterations = 1000
    clip = 1
    exp_cross = 0
    normalized_population, denorm_population = initialize_population(population_size, bounds)
    de1 = ClassicDE(bounds, mutation, cross_probability, normalized_population, denorm_population, population_size, iterations, clip, exp_cross, cost_function=lambda x: sum(x**2)/len(x))   # starting parameters given here
    result = list(de1.de())
    print(result[-1])

def initialize_population(population_size, bounds):
    dimensions = len(bounds)
    normalized_population = np.random.rand(population_size, dimensions)
    # check if bounds are given in proper order
    min_bound, max_bound = np.asarray(bounds).T
    bounds_difference = np.fabs(min_bound - max_bound)
    denorm_population = max_bound - bounds_difference * normalized_population
    return normalized_population, denorm_population



if __name__ == '__main__':
    main()
