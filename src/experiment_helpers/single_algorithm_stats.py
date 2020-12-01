import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def calculate_average_results(algorithm, xlabel, ylabel, figname, samples=25, **kwargs):
    all_results = pd.DataFrame()
    for i in tqdm(range(samples), leave=False):
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
