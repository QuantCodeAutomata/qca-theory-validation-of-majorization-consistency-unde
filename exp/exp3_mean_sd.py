import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.marginals import MEANS, SDS, SIGMA_C
from src.optimization import herfindahl_hirschman, solve_mean_sd
from src.utils import format_weights, print_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def validate_sigma_c():
    """Validate the comonotonic covariance matrix."""
    sigma1, sigma2, sigma3 = 2.6, 0.9, 0.2

    cov12 = sigma1 * sigma2  # = 2.34 under comonotonicity (Corr=1 for normal pairs)

    # For mixed normal-lognormal: Corr(Z, exp(Z)) = E[Z*exp(Z)] / (1 * SD(exp(Z)))
    # E[Z*exp(Z)] = e^(1/2) (by differentiation of MGF)
    # SD(exp(Z)) = sqrt(e^2 - e) = sqrt(e(e-1))
    # So Corr = e^(1/2) / sqrt(e(e-1)) = 1/sqrt(e-1)
    corr_z_expz = 1.0 / math.sqrt(math.e - 1)
    cov13 = sigma1 * sigma3 * corr_z_expz
    cov23 = sigma2 * sigma3 * corr_z_expz

    print(f'\nCovariance validation:')
    print(f'Cov(X1,X2) = {cov12:.4f} (paper: 2.3400)')
    print(f'Cov(X1,X3) = {cov13:.4f} (paper: 0.3967)')
    print(f'Cov(X2,X3) = {cov23:.4f} (paper: 0.1373)')
    print(f'Corr(Z,exp(Z)) = {corr_z_expz:.4f} (= 1/sqrt(e-1))')
    print(f'\nPaper Sigma_c:')
    print(SIGMA_C)


def solve_table4(kappa_values) -> pd.DataFrame:
    """Reproduce Table 4 at specified kappa values."""
    records = []
    for kappa in kappa_values:
        w, obj = solve_mean_sd(MEANS, SIGMA_C, kappa)
        V_SD = -obj  # sign flip: paper reports -J_c^*
        records.append({
            'kappa': kappa,
            'lambda_1': round(w[0], 4),
            'lambda_2': round(w[1], 4),
            'lambda_3': round(w[2], 4),
            'V_SD': round(V_SD, 4),
            'weights': format_weights(w)
        })
    df = pd.DataFrame(records)
    print_table(df, 'Table 4: Worst-Case Mean-SD Optimal Portfolios')
    return df


def compute_weight_path(kappa_grid: np.ndarray) -> np.ndarray:
    """Compute optimal weight path over kappa grid for Figure 1(c)."""
    weights_path = np.zeros((len(kappa_grid), 3))
    for i, kappa in enumerate(kappa_grid):
        w, _ = solve_mean_sd(MEANS, SIGMA_C, kappa)
        weights_path[i] = w
    return weights_path


def find_regime_boundaries(kappa_grid: np.ndarray, weights_path: np.ndarray) -> list:
    """Find kappa values where active asset changes."""
    dominant = np.argmax(weights_path, axis=1)
    boundaries = []
    for i in range(1, len(dominant)):
        if dominant[i] != dominant[i-1]:
            boundaries.append(kappa_grid[i])
    return boundaries


def plot_figure1c(kappa_grid, weights_path):
    """Generate Figure 1(c): optimal weight path vs kappa."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['steelblue', 'coral', 'green']
    for i in range(3):
        ax.plot(kappa_grid, weights_path[:, i], color=colors[i],
                linewidth=2, label=f'lambda_{i+1}')

    ax.set_xlabel('Risk Aversion kappa')
    ax.set_ylabel('Optimal Portfolio Weight')
    ax.set_title('Figure 1(c): Worst-Case Mean-SD Optimal Weights vs Kappa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(kappa_grid))
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp3_figure1c.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure 1(c) saved to {RESULTS_DIR}/exp3_figure1c.png')


def main():
    print('\n' + '='*60)
    print('EXPERIMENT 3: Worst-Case Mean-SD Optimization')
    print('='*60)

    validate_sigma_c()

    kappa_values = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25]
    df_t4 = solve_table4(kappa_values)

    print('\nExpected Table 4:')
    print('kappa=0.00: (1,0,0), V_SD=-4.0000')
    print('kappa=0.25: (1,0,0), V_SD=-4.6500')
    print('kappa=0.50: (0,1,0), V_SD=-5.2500')
    print('kappa=0.75: (0,0.2035,0.7965), V_SD=-5.4395')
    print('kappa=1.00: (0,0.0082,0.9918), V_SD=-5.4999')
    print('kappa=1.25: (0,0,1), V_SD=-5.5500')

    kappa_grid = np.linspace(0.001, 1.5, 300)
    weights_path = compute_weight_path(kappa_grid)

    boundaries = find_regime_boundaries(kappa_grid, weights_path)
    print(f'\nRegime boundaries (from optimization):')
    for b in boundaries:
        print(f'  kappa = {b:.4f}')
    print('Expected: ~0.4706, ~0.6690, ~1.0276')

    plot_figure1c(kappa_grid, weights_path)

    df_t4.to_csv(RESULTS_DIR / 'exp3_table4.csv', index=False)
    path_df = pd.DataFrame(weights_path, columns=['lambda_1', 'lambda_2', 'lambda_3'])
    path_df['kappa'] = kappa_grid
    path_df.to_csv(RESULTS_DIR / 'exp3_weight_path.csv', index=False)

    return df_t4


if __name__ == '__main__':
    main()
