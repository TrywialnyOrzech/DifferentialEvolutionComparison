import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.de import ClassicDE


def calculate_average_de_results(mutation, xlabel, ylabel, figname, bounds, normalized_population, denorm_population, population_size, cost_function, cross_probability, samples=25, exp=False):
    all_results = pd.DataFrame()
    for i in tqdm(range(samples), leave=False, desc='samples'):
        de = ClassicDE(
            bounds=bounds, mutation=mutation, cross_probability=cross_probability, normalized_population=normalized_population.copy(),
            denorm_population=denorm_population.copy(), population_size=population_size, iterations=2000,
            clip=1, exp_cross=exp, cost_function=cost_function
        )
        results = de.de()

        x, f = zip(*results)
        all_results[i] = f
        plt.plot(f)

    all_results['mean'] = all_results.mean(axis=1)

    plt.plot(all_results['mean'], color='black', linewidth=3, linestyle=':', label='Średnia')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figname)
    plt.close()

    return all_results


def calculate_average_results(algorithm, xlabel, ylabel, figname, samples=25, **kwargs):
    all_results = pd.DataFrame()
    for i in tqdm(range(samples), leave=False, desc='samples'):
        results = algorithm(**kwargs)

        x, f = zip(*results)
        all_results[i] = f
        plt.plot(f)

    all_results['mean'] = all_results.mean(axis=1)

    plt.plot(all_results['mean'], color='black', linewidth=3, linestyle=':', label='Średnia')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figname)
    plt.close()

    return all_results
