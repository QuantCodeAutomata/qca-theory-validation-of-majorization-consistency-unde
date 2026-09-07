import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization import solve_mean_sd, herfindahl_hirschman
from src.marginals import MEANS, SIGMA_C, SIGMA_0
from src.utils import format_weights, print_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def solve_both(kappa: float):
    """Solve mean-SD under independence and worst-case for a given kappa."""
    w0, j0 = solve_mean_sd(MEANS, SIGMA_0, kappa)
    wc, jc = solve_mean_sd(MEANS, SIGMA_C, kappa)
    return w0, j0, wc, jc


def compute_table5(kappa_values) -> pd.DataFrame:
    """Reproduce Table 5: comparison of independence vs worst-case portfolios."""
    records = []
    for kappa in kappa_values:
        w0, j0, wc, jc = solve_both(kappa)
        hhi0 = herfindahl_hirschman(w0)
        hhic = herfindahl_hirschman(wc)
        delta = jc - j0  # robustness premium (should be >= 0)
        records.append({
            'kappa': kappa,
            'lambda_0': format_weights(w0),
            'lambda_c': format_weights(wc),
            'HHI_0': round(hhi0, 4),
            'HHI_c': round(hhic, 4),
            'Delta': round(delta, 4),
        })
    df = pd.DataFrame(records)
    print_table(df, 'Table 5: Independence vs Worst-Case Portfolios')
    return df


def compute_paths(kappa_grid: np.ndarray):
    """Compute optimal weight paths for both dependence models."""
    path0 = np.zeros((len(kappa_grid), 3))
    pathc = np.zeros((len(kappa_grid), 3))
    hhi0_path = np.zeros(len(kappa_grid))
    hhic_path = np.zeros(len(kappa_grid))
    for i, kappa in enumerate(kappa_grid):
        w0, _, wc, _ = solve_both(kappa)
        path0[i] = w0
        pathc[i] = wc
        hhi0_path[i] = herfindahl_hirschman(w0)
        hhic_path[i] = herfindahl_hirschman(wc)
    return path0, pathc, hhi0_path, hhic_path


def plot_figure2ab(kappa_grid, path0, hhi0_path, hhic_path):
    """Generate Figure 2(a)-(b): weight paths and HHI comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure 2: Independence vs Worst-Case Dependence', fontsize=14, fontweight='bold')

    colors = ['steelblue', 'coral', 'green']
    ax = axes[0]
    for i in range(3):
        ax.plot(kappa_grid, path0[:, i], color=colors[i], linewidth=2, label=f'lambda_0_{i+1} (indep)')
    ax.set_xlabel('Risk Aversion kappa')
    ax.set_ylabel('Optimal Weight')
    ax.set_title('(a) Optimal Weights Under Independence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(kappa_grid[0], kappa_grid[-1])

    ax = axes[1]
    ax.plot(kappa_grid, hhi0_path, 'b-', linewidth=2, label='HHI (Independence)')
    ax.plot(kappa_grid, hhic_path, 'r--', linewidth=2, label='HHI (Worst-Case)')
    ax.set_xlabel('Risk Aversion kappa')
    ax.set_ylabel('HHI = sum(lambda_i^2)')
    ax.set_title('(b) Concentration (HHI) Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(kappa_grid[0], kappa_grid[-1])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp4_figure2ab.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure 2(a)-(b) saved to {RESULTS_DIR}/exp4_figure2ab.png')


def main():
    print('\n' + '='*60)
    print('EXPERIMENT 4: Value of Dependence Information')
    print('='*60)

    kappa_values = [0.50, 0.75, 1.00, 1.25]
    df_t5 = compute_table5(kappa_values)

    print('\nExpected Table 5:')
    print('kappa=0.50: HHI_0=0.5592, HHI_c=1.0000, Delta=0.1897')
    print('kappa=0.75: HHI_0=0.5612, HHI_c=0.6758, Delta=0.1372')
    print('kappa=1.00: HHI_0=0.5797, HHI_c=0.9838, Delta=0.0977')
    print('kappa=1.25: HHI_0=0.6672, HHI_c=1.0000, Delta=0.0832')

    print('\nComputing weight paths over kappa grid...')
    kappa_grid = np.linspace(0.01, 1.5, 200)
    path0, pathc, hhi0_path, hhic_path = compute_paths(kappa_grid)

    # Validate robustness premium is non-negative
    delta_path = np.array([solve_both(k)[3] - solve_both(k)[1] for k in kappa_grid])
    assert np.all(delta_path >= -1e-4), 'Robustness premium should be non-negative'
    print(f'Robustness premium is non-negative: True')

    plot_figure2ab(kappa_grid, path0, hhi0_path, hhic_path)

    df_t5.to_csv(RESULTS_DIR / 'exp4_table5.csv', index=False)
    path_df = pd.DataFrame({
        'kappa': kappa_grid,
        'lambda0_1': path0[:, 0], 'lambda0_2': path0[:, 1], 'lambda0_3': path0[:, 2],
        'lambdac_1': pathc[:, 0], 'lambdac_2': pathc[:, 1], 'lambdac_3': pathc[:, 2],
        'HHI_0': hhi0_path, 'HHI_c': hhic_path
    })
    path_df.to_csv(RESULTS_DIR / 'exp4_paths.csv', index=False)

    return df_t5


if __name__ == '__main__':
    main()
