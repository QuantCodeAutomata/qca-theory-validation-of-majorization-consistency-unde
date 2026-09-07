import sys
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.majorization import (
    apply_permutation,
    construct_beta_from_permutations,
    get_permutation_matrices,
    is_majorized,
    uniform_mixture_beta,
)
from src.marginals import MEANS, SDS, SIGMA_0
from src.risk_measures import normal_var, normal_es, portfolio_sd
from src.utils import format_weights, print_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def s_sd_exchangeable(weights: np.ndarray, sigma: float = 1.0) -> float:
    """S_SD(lambda; X) for X iid N(0, sigma^2). = sigma * ||lambda||_2."""
    return sigma * float(np.linalg.norm(weights, 2))


def s_es_exchangeable(weights: np.ndarray, p: float = 0.95, sigma: float = 1.0) -> float:
    """S_ES(lambda; X) for X iid N(0, sigma^2). = sigma * ||lambda||_2 * phi(Phi^{-1}(p))/(1-p)."""
    z = stats.norm.ppf(p)
    cp = stats.norm.pdf(z) / (1 - p)
    return sigma * float(np.linalg.norm(weights, 2)) * cp


def s_sd_nonexchangeable(weights: np.ndarray, cov: np.ndarray) -> float:
    """S_SD(lambda; X) = sqrt(lambda^T Sigma lambda) for general covariance."""
    return portfolio_sd(weights, cov)


def run_exchangeable_validation() -> pd.DataFrame:
    """Validate under iid N(0,1) (exchangeable): S_rho(beta) <= S_rho(lambda)."""
    n = 3
    test_cases = [
        {
            'name': 'concentrate -> uniform',
            'lambda': np.array([1.0, 0.0, 0.0]),
            'beta': np.array([1/3, 1/3, 1/3])
        },
        {
            'name': '(0.7,0.2,0.1) -> uniform avg',
            'lambda': np.array([0.7, 0.2, 0.1]),
            'beta': None  # Will be uniform mixture
        },
        {
            'name': '(0.6,0.3,0.1) -> (0.5,0.3,0.2)',
            'lambda': np.array([0.6, 0.3, 0.1]),
            'beta': np.array([0.5, 0.3, 0.2])
        },
        {
            'name': '(0.8,0.1,0.1) -> (0.4,0.4,0.2)',
            'lambda': np.array([0.8, 0.1, 0.1]),
            'beta': np.array([0.4, 0.4, 0.2])
        }
    ]

    records = []
    perm_mats = get_permutation_matrices(n)

    for tc in test_cases:
        lam = tc['lambda']
        if tc['beta'] is None:
            beta, permuted = uniform_mixture_beta(lam)
        else:
            beta = tc['beta']

        assert is_majorized(beta, lam), f"beta not majorized by lambda in case {tc['name']}"

        for rho_name, rho_fn in [
            ('SD', s_sd_exchangeable),
            ('ES_0.95', lambda w: s_es_exchangeable(w, p=0.95))
        ]:
            s_beta = rho_fn(beta)
            s_perm = [rho_fn(P @ lam) for P in perm_mats]
            s_lambda = rho_fn(lam)
            max_perm = max(s_perm)

            weak_ok = s_beta <= max_perm + 1e-8
            strong_ok = s_beta <= s_lambda + 1e-8  # holds under exchangeability

            records.append({
                'Case': tc['name'],
                'rho': rho_name,
                'lambda': format_weights(lam),
                'beta': format_weights(beta),
                'S_rho(beta)': round(s_beta, 6),
                'S_rho(lambda)': round(s_lambda, 6),
                'max_perm_risk': round(max_perm, 6),
                'Weak_OK': weak_ok,
                'Strong_OK': strong_ok
            })

    df = pd.DataFrame(records)
    print_table(df, 'Exp 1: Exchangeable Validation (iid N(0,1))')
    return df


def run_nonexchangeable_validation() -> pd.DataFrame:
    """Validate under non-exchangeable Section 7 marginals (independence) using SD."""
    n = 3
    Sigma = SIGMA_0  # diagonal independence covariance

    test_cases = [
        {
            'name': 'e1 -> uniform',
            'lambda': np.array([1.0, 0.0, 0.0]),
            'beta': np.array([1/3, 1/3, 1/3])
        },
        {
            'name': '(0.7,0.2,0.1) -> uniform avg',
            'lambda': np.array([0.7, 0.2, 0.1]),
            'beta': None  # uniform mixture
        },
        {
            'name': '(0.6,0.3,0.1) -> (0.5,0.3,0.2)',
            'lambda': np.array([0.6, 0.3, 0.1]),
            'beta': np.array([0.5, 0.3, 0.2])
        },
    ]

    records = []
    perm_mats = get_permutation_matrices(n)

    for tc in test_cases:
        lam = tc['lambda']
        if tc['beta'] is None:
            beta, _ = uniform_mixture_beta(lam)
        else:
            beta = tc['beta']

        assert is_majorized(beta, lam), f'beta not majorized by lambda: {tc["name"]}'

        s_beta = s_sd_nonexchangeable(beta, Sigma)
        s_perm = [s_sd_nonexchangeable(P @ lam, Sigma) for P in perm_mats]
        s_lambda = s_sd_nonexchangeable(lam, Sigma)
        max_perm = max(s_perm)

        weak_ok = s_beta <= max_perm + 1e-8

        records.append({
            'Case': tc['name'],
            'lambda': format_weights(lam),
            'beta': format_weights(beta),
            'S_SD(beta)': round(s_beta, 6),
            'S_SD(lambda)': round(s_lambda, 6),
            'max_perm_risk': round(max_perm, 6),
            'Weak_OK': weak_ok,
        })

    df = pd.DataFrame(records)
    print_table(df, 'Exp 1: Non-Exchangeable Validation (Section 7 Independence)')
    return df


def run_s_rho_max_monotonicity() -> pd.DataFrame:
    """Verify monotonicity of S_rho^max(lambda) = max_k S_rho(Pi_k lambda) along majorization order."""
    n = 3
    perm_mats = get_permutation_matrices(n)
    e1 = np.array([1.0, 0.0, 0.0])
    uniform = np.array([1/3, 1/3, 1/3])

    alphas = np.linspace(0, 1, 11)
    records = []
    for a in alphas:
        lam = (1 - a) * e1 + a * uniform
        lam /= lam.sum()
        s_max_sd = max(s_sd_exchangeable(P @ lam) for P in perm_mats)
        s_max_es = max(s_es_exchangeable(P @ lam, p=0.95) for P in perm_mats)
        records.append({
            'alpha': round(a, 2),
            'lambda': format_weights(lam),
            'S_SD^max': round(s_max_sd, 6),
            'S_ES^max': round(s_max_es, 6),
        })

    df = pd.DataFrame(records)
    print_table(df, 'Exp 1: S_rho^max Monotonicity Along Majorization Path')
    return df


def plot_results(df_exch, df_nonexch, df_mono):
    """Generate and save visualization for Experiment 1."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Exp 1: Majorization Consistency Validation', fontsize=14, fontweight='bold')

    # Panel 1: Exchangeable - S_rho(beta) vs max permutation risk
    ax = axes[0]
    sd_rows = df_exch[df_exch['rho'] == 'SD']
    x = range(len(sd_rows))
    ax.bar([xi - 0.2 for xi in x], sd_rows['S_rho(beta)'].values, width=0.35,
           label='S_SD(beta)', color='steelblue')
    ax.bar([xi + 0.2 for xi in x], sd_rows['max_perm_risk'].values, width=0.35,
           label='max S_SD(Pi_k lambda)', color='coral')
    ax.set_xticks(list(x))
    ax.set_xticklabels([f'Case {i+1}' for i in x], rotation=15, ha='right')
    ax.set_title('Exchangeable: SD (Weak Consistency)')
    ax.set_ylabel('Portfolio SD')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Non-exchangeable weak consistency
    ax = axes[1]
    x2 = range(len(df_nonexch))
    ax.bar([xi - 0.2 for xi in x2], df_nonexch['S_SD(beta)'].values, width=0.35,
           label='S_SD(beta)', color='steelblue')
    ax.bar([xi + 0.2 for xi in x2], df_nonexch['max_perm_risk'].values, width=0.35,
           label='max S_SD(Pi_k lambda)', color='coral')
    ax.set_xticks(list(x2))
    ax.set_xticklabels([f'Case {i+1}' for i in x2], rotation=15, ha='right')
    ax.set_title('Non-Exchangeable: SD (Weak Consistency)')
    ax.set_ylabel('Portfolio SD')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: S_rho^max monotonicity
    ax = axes[2]
    ax.plot(df_mono['alpha'], df_mono['S_SD^max'], 'o-', color='steelblue', label='S_SD^max')
    ax.plot(df_mono['alpha'], df_mono['S_ES^max'], 's--', color='coral', label='S_ES^max')
    ax.set_xlabel('alpha (0=e1, 1=uniform)')
    ax.set_ylabel('S_rho^max(lambda)')
    ax.set_title('S_rho^max Monotonicity Along Path')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp1_majorization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {RESULTS_DIR}/exp1_majorization.png')


def main():
    print('\n' + '='*60)
    print('EXPERIMENT 1: Majorization Consistency Theory Validation')
    print('='*60)

    df_exch = run_exchangeable_validation()
    df_nonexch = run_nonexchangeable_validation()
    df_mono = run_s_rho_max_monotonicity()

    all_weak_ok = df_exch['Weak_OK'].all() and df_nonexch['Weak_OK'].all()
    all_strong_ok = df_exch['Strong_OK'].all()

    print(f'\nAll weak consistency checks passed: {all_weak_ok}')
    print(f'All strong consistency checks (exchangeable) passed: {all_strong_ok}')

    df_exch.to_csv(RESULTS_DIR / 'exp1_exchangeable.csv', index=False)
    df_nonexch.to_csv(RESULTS_DIR / 'exp1_nonexchangeable.csv', index=False)
    df_mono.to_csv(RESULTS_DIR / 'exp1_monotonicity.csv', index=False)

    plot_results(df_exch, df_nonexch, df_mono)

    return df_exch, df_nonexch, df_mono


if __name__ == '__main__':
    main()
