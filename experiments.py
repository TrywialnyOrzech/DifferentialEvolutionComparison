from tqdm import tqdm

from src.cost_functions import rosenbrock, levy, michalewicz, bowl
from src.experiment_helpers.whole_tests import run_experiments_for_cost_function


def run():
    cases = [
        {
            'cost_function': rosenbrock,
            'cost_function_name': 'rosenbrock',
            'bounds': [(-5, 10)]
        },
        {
            'cost_function': bowl,
            'cost_function_name': 'bowl',
            'bounds': [(-100, 100)]
        },
        {
            'cost_function': levy,
            'cost_function_name': 'levy',
            'bounds': [(-10, 10)]
        },
        {
            'cost_function': michalewicz,
            'cost_function_name': 'michalewicz',
            'bounds': [(0, 3.14159265359)]  # 0 to π
        },
    ]

    for case in tqdm(cases):
        de_results, bbde_results = run_experiments_for_cost_function(
            cost_function=case['cost_function'],
            cost_function_name=case['cost_function_name'],
            iterations=2000,
            bounds=case['bounds'],
        )
        bbde_results.to_csv(f"bbde-{case['cost_function_name']}")
        de_results.to_csv(f"de-{case['cost_function_name']}")


if __name__ == '__main__':
    run()
