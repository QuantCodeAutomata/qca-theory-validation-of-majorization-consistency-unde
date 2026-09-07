"""Master runner: executes all 8 experiments and saves aggregated results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import traceback

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_experiment(name: str, fn):
    print(f'\n{"-"*70}')
    print(f'Running {name}...')
    print(f'{"-"*70}')
    try:
        result = fn()
        print(f'[OK] {name} completed successfully.')
        return True, result
    except Exception as e:
        print(f'[FAILED] {name} raised an error:')
        traceback.print_exc()
        return False, None


def main():
    from exp.exp1_majorization import main as exp1_main
    from exp.exp2_tail_risk import main as exp2_main
    from exp.exp3_mean_sd import main as exp3_main
    from exp.exp4_dependence_info import main as exp4_main
    from exp.exp5_tail_shape import main as exp5_main
    from exp.exp6_counterexample import main as exp6_main
    from exp.exp7_structured_uncertainty import main as exp7_main
    from exp.exp8_weighted_robustness import main as exp8_main

    experiments = [
        ('Exp 1: Majorization Consistency', exp1_main),
        ('Exp 2: Tail-Risk Concentration', exp2_main),
        ('Exp 3: Worst-Case Mean-SD', exp3_main),
        ('Exp 4: Value of Dependence Information', exp4_main),
        ('Exp 5: Tail-Shape Sensitivity', exp5_main),
        ('Exp 6: Counterexample (Diversification Under SD)', exp6_main),
        ('Exp 7: Structured Uncertainty Reductions', exp7_main),
        ('Exp 8: Weighted Robustness', exp8_main),
    ]

    print('='*70)
    print('RUNNING ALL EXPERIMENTS')
    print('='*70)

    statuses = []
    for name, fn in experiments:
        ok, _ = run_experiment(name, fn)
        statuses.append((name, 'PASSED' if ok else 'FAILED'))

    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    for name, status in statuses:
        print(f'  [{status}] {name}')

    n_passed = sum(1 for _, s in statuses if s == 'PASSED')
    print(f'\n{n_passed}/{len(statuses)} experiments completed successfully.')

    return statuses


if __name__ == '__main__':
    main()
