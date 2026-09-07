import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, integrate
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import print_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def laplace_quantile(u: float) -> float:
    """Quantile function of Laplace(0, 1/sqrt(2)) (mean=0, variance=1).
    F^{-1}(u) = -(1/sqrt(2)) * sign(u-0.5) * ln(1 - 2|u-0.5|)
    """
    b = 1.0 / np.sqrt(2.0)
    v = u - 0.5
    return -b * np.sign(v) * np.log(1.0 - 2.0 * np.abs(v))


def normal_quantile(u: float) -> float:
    """Quantile function of N(0,1)."""
    return stats.norm.ppf(u)


def compute_covariance_comonotonic(eps: float = 1e-10) -> float:
    """Compute c = Cov(F1^{-1}(U), F2^{-1}(U)) = integral_eps^{1-eps} F1^{-1}(u)*F2^{-1}(u) du.
    Since E[F1^{-1}(U)] = E[F2^{-1}(U)] = 0, the covariance equals the integral of the product.
    Splits integral at 0.5 to handle the kink in the Laplace quantile.
    """
    def integrand(u):
        return normal_quantile(u) * laplace_quantile(u)

    # Split at 0.5 for accuracy near the Laplace kink
    result1, err1 = integrate.quad(integrand, eps, 0.5, limit=500, epsabs=1e-10, epsrel=1e-10)
    result2, err2 = integrate.quad(integrand, 0.5, 1 - eps, limit=500, epsabs=1e-10, epsrel=1e-10)
    c = result1 + result2
    return c


def compute_portfolio_variance(lam1: float, lam2: float, c: float) -> float:
    """Var(lam1*F1^{-1}(U) + lam2*F2^{-1}(U)) = lam1^2*1 + lam2^2*1 + 2*lam1*lam2*c."""
    return lam1**2 + lam2**2 + 2 * lam1 * lam2 * c


def scan_simplex(c: float, n_points: int = 101) -> pd.DataFrame:
    """Scan over all lambda = (t, 1-t) for t in [0,1] and compute worst-case variance."""
    records = []
    for t in np.linspace(0, 1, n_points):
        var_wc = compute_portfolio_variance(t, 1 - t, c)
        records.append({'lambda_1': round(t, 2), 'lambda_2': round(1-t, 2), 'WC_Variance': round(var_wc, 6)})
    return pd.DataFrame(records)


def main():
    print('\n' + '='*60)
    print('EXPERIMENT 6: Counterexample - Diversification Under Worst-Case SD')
    print('='*60)

    print('\nAsset 1: X1 ~ N(0,1), Var=1')
    print('Asset 2: X2 ~ Laplace(0, 1/sqrt(2)), Var=1')
    print('Worst-case coupling: comonotonic (common U ~ Uniform[0,1])')

    # Compute comonotonic covariance
    print('\nComputing Cov(F1^{-1}(U), F2^{-1}(U))...')
    c = compute_covariance_comonotonic()
    print(f'c = Cov(F1^{{-1}}(U), F2^{{-1}}(U)) = {c:.8f}')
    assert c < 1.0, f'Expected c < 1, got {c}'
    print(f'c < 1: True (confirming diversification benefit)')

    var_conc1 = compute_portfolio_variance(1.0, 0.0, c)
    var_conc2 = compute_portfolio_variance(0.0, 1.0, c)
    var_div = compute_portfolio_variance(0.5, 0.5, c)

    expected_div_var = (1 + c) / 2

    print(f'\nVar(1,0): {var_conc1:.8f} (expected: 1.0)')
    print(f'Var(0,1): {var_conc2:.8f} (expected: 1.0)')
    print(f'Var(0.5,0.5): {var_div:.8f}')
    print(f'(1+c)/2 = {expected_div_var:.8f}')
    print(f'Diversified variance < 1: {var_div < 1.0}')
    assert abs(var_div - expected_div_var) < 1e-8
    assert var_div < var_conc1
    assert var_div < var_conc2

    df_simplex = scan_simplex(c)

    results = pd.DataFrame({
        'Portfolio': ['(1,0)', '(0,1)', '(0.5,0.5)'],
        'lambda_1': [1.0, 0.0, 0.5],
        'lambda_2': [0.0, 1.0, 0.5],
        'WC_Variance': [var_conc1, var_conc2, var_div],
        'WC_SD': [np.sqrt(var_conc1), np.sqrt(var_conc2), np.sqrt(var_div)]
    })
    print_table(results, 'Exp 6: Worst-Case Variance Comparison')
    print(f'\nc = {c:.6f}')
    print(f'(1+c)/2 = {expected_div_var:.6f} < 1: diversification is risk-reducing!')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Exp 6: Counterexample - Normal vs Laplace Under Comonotonicity', fontsize=13)

    ax = axes[0]
    ax.plot(df_simplex['lambda_1'], df_simplex['WC_Variance'], 'b-', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Concentrated variance=1')
    ax.axvline(x=0.5, color='g', linestyle=':', label='Diversified (0.5,0.5)')
    ax.set_xlabel('lambda_1 (weight on N(0,1))')
    ax.set_ylabel('Worst-Case Variance')
    ax.set_title('Worst-Case Variance over Simplex')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    u_grid = np.linspace(0.001, 0.999, 500)
    q1 = stats.norm.ppf(u_grid)
    q2 = np.array([laplace_quantile(u) for u in u_grid])
    ax.plot(u_grid, q1, 'b-', linewidth=2, label='N(0,1) quantile')
    ax.plot(u_grid, q2, 'r--', linewidth=2, label='Laplace quantile')
    ax.set_xlabel('u')
    ax.set_ylabel('F^{-1}(u)')
    ax.set_title('Quantile Functions (Comonotonic Coupling)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp6_counterexample.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {RESULTS_DIR}/exp6_counterexample.png')

    results.to_csv(RESULTS_DIR / 'exp6_results.csv', index=False)
    pd.DataFrame({'c': [c], 'diversified_variance': [var_div], 'formula_check': [expected_div_var]}).to_csv(
        RESULTS_DIR / 'exp6_covariance.csv', index=False
    )

    return results


if __name__ == '__main__':
    main()
