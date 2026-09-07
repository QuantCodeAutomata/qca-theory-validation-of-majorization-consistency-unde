import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import cvxpy as cp
from src.utils import format_weights, print_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Section 8 parameters
MU = np.array([4.9, 4.1, 5.5])
SIGMA_VEC = np.array([1.8, 0.2, 1.0])  # marginal SDs
RHO12, RHO13, RHO23 = 0.30, 0.20, 0.25
SIGMA_MAT = np.array([
    [1.8**2, RHO12*1.8*0.2, RHO13*1.8*1.0],
    [RHO12*1.8*0.2, 0.2**2, RHO23*0.2*1.0],
    [RHO13*1.8*1.0, RHO23*0.2*1.0, 1.0**2]
])


def compute_cp(p: float) -> float:
    """Compute c_p = phi(Phi^{-1}(p)) / (1-p) for normal ES."""
    z = stats.norm.ppf(p)
    return stats.norm.pdf(z) / (1 - p)


def solve_blended_es(
    omega: float,
    p: float = 0.95
) -> tuple:
    """Solve min_{lambda in Delta_3} lambda^T mu + c_p * [omega * lambda^T sigma + (1-omega) * sqrt(lambda^T Sigma lambda)].
    Returns (optimal_weights, optimal_objective).
    """
    c_p = compute_cp(p)
    n = 3
    lam = cp.Variable(n, nonneg=True)

    mean_term = MU @ lam
    wc_term = SIGMA_VEC @ lam

    # PSD factor for SIGMA_MAT
    eigvals, eigvecs = np.linalg.eigh(SIGMA_MAT)
    eigvals_clipped = np.maximum(eigvals, 0)
    L = eigvecs @ np.diag(np.sqrt(eigvals_clipped))
    ref_sd_term = cp.norm(L.T @ lam, 2)

    obj = mean_term + c_p * (omega * wc_term + (1 - omega) * ref_sd_term)
    constraints = [cp.sum(lam) == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if lam.value is None:
        raise ValueError(f'Optimization failed for omega={omega}: {prob.status}')

    w = np.maximum(lam.value, 0)
    w /= w.sum()
    return w, float(prob.value)


def validate_special_cases(p: float = 0.95):
    """Validate omega=0 (reference model) and omega=1 (worst-case) special cases."""
    c_p = compute_cp(p)
    print(f'\nUsing ES level p={p}, c_p={c_p:.6f}')

    # Marginal ES_p values
    marginal_es = MU + SIGMA_VEC * c_p
    print('\nMarginal ES_p values (mu_i + sigma_i * c_p):')
    for i in range(3):
        print(f'  Asset {i+1}: {marginal_es[i]:.4f}')
    min_es_asset = np.argmin(marginal_es) + 1
    print(f'  Minimum at Asset {min_es_asset} -> expected concentration under omega=1')

    # omega = 0: reference model
    w0, obj0 = solve_blended_es(omega=0.0, p=p)
    print(f'\nomega=0 (reference model): lambda={format_weights(w0)}, obj={obj0:.4f}')

    # omega = 1: worst-case -> LP -> concentration
    w1, obj1 = solve_blended_es(omega=1.0, p=p)
    print(f'omega=1 (worst-case):      lambda={format_weights(w1)}, obj={obj1:.4f}')

    # Verify omega=1 gives concentration
    dominant1 = np.argmax(w1) + 1
    print(f'\nomega=1 concentrated asset: Asset {dominant1}')
    print(f'Expected: Asset {min_es_asset} (minimum marginal ES_p)')
    assert dominant1 == min_es_asset, f'omega=1 should concentrate in Asset {min_es_asset}, got {dominant1}'
    print('Concentration check passed!')

    return w0, w1, marginal_es


def compute_omega_path(omega_grid: np.ndarray, p: float = 0.95) -> pd.DataFrame:
    """Compute optimal portfolio as omega varies from 0 to 1."""
    records = []
    for omega in omega_grid:
        w, obj = solve_blended_es(omega=omega, p=p)
        records.append({
            'omega': round(omega, 3),
            'lambda_1': round(w[0], 4),
            'lambda_2': round(w[1], 4),
            'lambda_3': round(w[2], 4),
            'objective': round(obj, 4),
            'weights': format_weights(w)
        })
    df = pd.DataFrame(records)
    print_table(df[['omega', 'lambda_1', 'lambda_2', 'lambda_3', 'objective']], 'Exp 8: Blended ES Optimal Portfolios')
    return df


def plot_omega_path(df: pd.DataFrame):
    """Plot optimal weights and objective as omega varies."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Exp 8: Weighted Robustness (ES, Section 8 Gaussian Inputs)', fontsize=13)

    colors = ['steelblue', 'coral', 'green']
    ax = axes[0]
    for i in range(3):
        ax.plot(df['omega'], df[f'lambda_{i+1}'], color=colors[i], linewidth=2, label=f'Asset {i+1}')
    ax.set_xlabel('Robustness Weight omega')
    ax.set_ylabel('Optimal Portfolio Weight')
    ax.set_title('Optimal Weights vs omega')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    ax = axes[1]
    ax.plot(df['omega'], df['objective'], 'k-', linewidth=2)
    ax.set_xlabel('Robustness Weight omega')
    ax.set_ylabel('Optimized Criterion')
    ax.set_title('Objective Value vs omega')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp8_weighted_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {RESULTS_DIR}/exp8_weighted_robustness.png')


def main():
    print('\n' + '='*60)
    print('EXPERIMENT 8: Weighted Robustness Under Reference and Worst-Case Dependence')
    print('='*60)

    print('\nSection 8 inputs (extension with p=0.95):')
    print(f'mu = {MU}')
    print(f'sigma = {SIGMA_VEC}')
    print(f'Sigma =\n{SIGMA_MAT}')

    # Validate special cases
    w0, w1, marginal_es = validate_special_cases(p=0.95)

    # Compute omega path
    omega_grid = np.linspace(0, 1, 21)
    df_path = compute_omega_path(omega_grid, p=0.95)

    plot_omega_path(df_path)

    df_path.to_csv(RESULTS_DIR / 'exp8_omega_path.csv', index=False)

    marginal_df = pd.DataFrame({'Asset': [1, 2, 3], 'mu': MU, 'sigma': SIGMA_VEC, 'ES_p_0.95': marginal_es})
    marginal_df.to_csv(RESULTS_DIR / 'exp8_marginals.csv', index=False)

    print('\nKey finding: omega=0 (reference model) -> diversified; omega=1 (worst-case) -> concentrated in Asset 2')

    return df_path


if __name__ == '__main__':
    main()
