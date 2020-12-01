import numpy as np
import pandas as pd
from tqdm import tqdm

from src import bbde
from src.de import ClassicDE
from src.experiment_helpers.single_algorithm_stats import calculate_average_results


def run_experiments_for_cost_function(cost_function, cost_function_name, iterations, bounds, dimensions=10, **kwargs):
    populations = [20, 40, 60, 80, 100, 200]

    all_de_results = pd.DataFrame()
    all_bbde_results = pd.DataFrame()
    for popsize in tqdm(populations, leave=False):

        pop = np.random.rand(popsize, dimensions)
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        population = min_b + pop * diff  # denormalisation

        de = ClassicDE()
        de_results = calculate_average_results(

        )

        bbde_results = calculate_average_results(
            bbde.bbde,
            xlabel='Iteracje',
            ylabel='Wartość funkcji kosztu',
            figname=f'{cost_function_name}-pop{pop}.png',
            fobj=cost_function,
            bounds=bounds * dimensions,
            its=iterations,
            population=population,
            samples=25,
        )
        all_bbde_results[f'{pop}'] = bbde_results['mean']

    return all_de_results, all_bbde_results
