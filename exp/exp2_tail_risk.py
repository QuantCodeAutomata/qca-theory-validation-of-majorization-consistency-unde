import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.marginals import MEANS, SDS, asset3_params
from src.risk_measures import (
    lognormal_rvar,
    lognormal_var,
    normal_rvar,
    normal_var,
)
from src.utils import print_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Asset 3 lognormal parameters: X3 = a3 + b3*exp(Z), Z ~ N(0,1)
a3, b3 = asset3_params(eta=1.0)

# Risk measure parameters
VAR_P = 0.99
RVAR_BETA = 0.01
RVAR_ALPHA = 0.10


def compute_table2() -> pd.DataFrame:
    """Compute Table 2: marginal statistics for 3 assets."""
    var_vals = [
        normal_var(4.0, 2.6, VAR_P),
        normal_var(4.8, 0.9, VAR_P),
        lognormal_var(a3, b3, VAR_P)
    ]
    rvar_vals = [
        normal_rvar(4.0, 2.6, RVAR_BETA, RVAR_ALPHA),
        normal_rvar(4.8, 0.9, RVAR_BETA, RVAR_ALPHA),
        lognormal_rvar(a3, b3, RVAR_BETA, RVAR_ALPHA)
    ]
    df = pd.DataFrame({
        'Asset': [1, 2, 3],
        'Mean': MEANS,
        'SD': SDS,
        'VaR_0.99': [round(v, 4) for v in var_vals],
        'RVaR_0.01_0.10': [round(v, 4) for v in rvar_vals]
    })
    print_table(df, 'Table 2: Marginal Asset Statistics')
    return df


def compute_switching_thresholds(
    means: np.ndarray,
    risks: np.ndarray
) -> dict:
    """Compute pairwise switching thresholds kappa_ij = (m_j - m_i) / (r_i - r_j).
    Returns dict of threshold values and regime structure.
    """
    n = len(means)
    thresholds = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                denom = risks[i] - risks[j]
                if abs(denom) > 1e-10:
                    kappa_ij = (means[j] - means[i]) / denom
                    thresholds[f'kappa_{i+1}{j+1}'] = kappa_ij
    return thresholds


def compute_affine_objectives(
    means: np.ndarray,
    risks: np.ndarray,
    kappa_grid: np.ndarray
) -> np.ndarray:
    """Compute f_i^rho(kappa) = -m_i - kappa * r_i^rho for each asset i.
    Returns array of shape (len(kappa_grid), n_assets).
    """
    n = len(means)
    F = np.zeros((len(kappa_grid), n))
    for i in range(n):
        F[:, i] = -means[i] - kappa_grid * risks[i]
    return F


def compute_optimal_asset(F: np.ndarray) -> np.ndarray:
    """Return index of asset achieving upper envelope (argmax over assets)."""
    return np.argmax(F, axis=1)


def find_regime_boundaries(F: np.ndarray, kappa_grid: np.ndarray) -> list:
    """Find kappa values where optimal asset changes."""
    optimal = compute_optimal_asset(F)
    boundaries = []
    for i in range(1, len(optimal)):
        if optimal[i] != optimal[i-1]:
            boundaries.append(kappa_grid[i])
    return boundaries


def compute_table3(means, var_risks, rvar_risks):
    """Reproduce Table 3: regime structure for VaR and RVaR."""
    kappa_grid = np.linspace(0, 2.0, 100000)
    F_var = compute_affine_objectives(means, var_risks, kappa_grid)
    F_rvar = compute_affine_objectives(means, rvar_risks, kappa_grid)

    opt_var = compute_optimal_asset(F_var)
    opt_rvar = compute_optimal_asset(F_rvar)

    var_boundaries = find_regime_boundaries(F_var, kappa_grid)
    rvar_boundaries = find_regime_boundaries(F_rvar, kappa_grid)

    print('\n=== Table 3: Regime Structure ===')
    print('\nVaR_0.99^+ regimes:')
    print(f'  e_1 optimal for kappa in [0, {var_boundaries[0]:.4f}]')
    print(f'  e_2 optimal for kappa in [{var_boundaries[0]:.4f}, {var_boundaries[1]:.4f}]')
    print(f'  e_3 optimal for kappa > {var_boundaries[1]:.4f}')

    print('\nRVaR regimes:')
    print(f'  e_1 optimal for kappa in [0, {rvar_boundaries[0]:.4f}]')
    print(f'  e_2 optimal for kappa in [{rvar_boundaries[0]:.4f}, {rvar_boundaries[1]:.4f}]')
    print(f'  e_3 optimal for kappa > {rvar_boundaries[1]:.4f}')

    return var_boundaries, rvar_boundaries


def plot_figure1(means, var_risks, rvar_risks):
    """Generate Figure 1(a)-(b): line envelope plots."""
    kappa_grid = np.linspace(0, 2.0, 2000)

    F_var = compute_affine_objectives(means, var_risks, kappa_grid)
    F_rvar = compute_affine_objectives(means, rvar_risks, kappa_grid)

    env_var = np.max(F_var, axis=1)
    env_rvar = np.max(F_rvar, axis=1)

    colors = ['steelblue', 'coral', 'green']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 1: Affine Objective Lines and Upper Envelope',
                 fontsize=14, fontweight='bold')

    for i in range(3):
        axes[0].plot(
            kappa_grid, F_var[:, i], color=colors[i], linewidth=1.5,
            label=f'Asset {i+1}: f={-means[i]:.2f}-{var_risks[i]:.4f}*kappa'
        )
        axes[1].plot(
            kappa_grid, F_rvar[:, i], color=colors[i], linewidth=1.5,
            label=f'Asset {i+1}: f={-means[i]:.2f}-{rvar_risks[i]:.4f}*kappa'
        )

    axes[0].plot(kappa_grid, env_var, 'k--', linewidth=2, label='Upper Envelope')
    axes[1].plot(kappa_grid, env_rvar, 'k--', linewidth=2, label='Upper Envelope')

    for ax, title in zip(axes, ['(a) VaR_0.99^+', '(b) RVaR_{0.01,0.10}']):
        ax.set_xlabel('Risk Aversion kappa')
        ax.set_ylabel('f_i(kappa) = -m_i - kappa * r_i')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.0)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp2_figure1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure 1 saved to {RESULTS_DIR}/exp2_figure1.png')


def main():
    print('\n' + '='*60)
    print('EXPERIMENT 2: Tail-Risk Concentration Under Dependence Ambiguity')
    print('='*60)

    df_t2 = compute_table2()

    means = MEANS
    var_risks = df_t2['VaR_0.99'].values
    rvar_risks = df_t2['RVaR_0.01_0.10'].values

    print('\nExpected (Table 2):')
    print('Asset 1: 10.0485, 8.1960')
    print('Asset 2:  6.8937, 6.2525')
    print('Asset 3:  6.0951, 5.6327')

    var_boundaries, rvar_boundaries = compute_table3(means, var_risks, rvar_risks)

    print('\nExpected thresholds:')
    print('VaR:  kappa_12=0.2536, kappa_23=0.6261')
    print('RVaR: kappa_12=0.4116, kappa_23=0.8067')

    plot_figure1(means, var_risks, rvar_risks)

    df_t2.to_csv(RESULTS_DIR / 'exp2_table2.csv', index=False)
    regime_df = pd.DataFrame({
        'Risk Measure': ['VaR_0.99^+', 'RVaR_{0.01,0.10}'],
        'kappa_12': [var_boundaries[0], rvar_boundaries[0]],
        'kappa_23': [var_boundaries[1], rvar_boundaries[1]]
    })
    regime_df.to_csv(RESULTS_DIR / 'exp2_table3_regimes.csv', index=False)
    print_table(regime_df, 'Table 3: Switching Thresholds')

    return df_t2, regime_df


if __name__ == '__main__':
    main()
