import numpy as np
import pandas as pd
from tqdm import tqdm

from src import bbde
from src.de import ClassicDE
from src.experiment_helpers.single_algorithm_stats import calculate_average_results


def run_experiments_for_cost_function(cost_function, cost_function_name, iterations, bounds, dimensions=10):
    populations = [20, 40, 60, 80, 100, 120]

    all_de_results = pd.DataFrame()
    all_bbde_results = pd.DataFrame()
    for popsize in tqdm(populations, leave=False):

        normalized_population, denorm_population = initialize_population(popsize, bounds * dimensions)

        # de = ClassicDE(bounds)
        # de_results = calculate_average_results(
        #     de.de,
        #     xlabel='Iteracje',
        #     ylabel='Wartość funkcji kosztu',
        #     figname=f'de-{cost_function_name}-{popsize}.png',
        #     fobj=cost_function,
        #     samples=25,
        # )

        bbde_results = calculate_average_results(
            bbde.bbde,
            xlabel='Iteracje',
            ylabel='Wartość funkcji kosztu',
            figname=f'bbde-{cost_function_name}-alt-{popsize}.png',
            fobj=cost_function,
            samples=25,
            its=iterations,
            pop=denorm_population.copy(),
            alternative_exp_offset=True,
        )
        all_bbde_results[f'alt{popsize}'] = bbde_results['mean']

        bbde_results = calculate_average_results(
            bbde.bbde,
            xlabel='Iteracje',
            ylabel='Wartość funkcji kosztu',
            figname=f'bbde-{cost_function_name}-nor-{popsize}.png',
            fobj=cost_function,
            samples=25,
            its=iterations,
            pop=denorm_population.copy(),
            alternative_exp_offset=False,
        )
        all_bbde_results[f'nor{popsize}'] = bbde_results['mean']

    return all_de_results, all_bbde_results


def initialize_population(population_size, bounds):
    dimensions = len(bounds)
    normalized_population = np.random.rand(population_size, dimensions)
    # check if bounds are given in proper order
    min_bound, max_bound = np.asarray(bounds).T
    bounds_difference = np.fabs(min_bound - max_bound)
    denorm_population = max_bound - bounds_difference * normalized_population
    return normalized_population, denorm_population
