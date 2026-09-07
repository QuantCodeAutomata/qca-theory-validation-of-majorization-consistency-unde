import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, integrate, optimize
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import print_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def distortion_cvar(t: float, p: float) -> float:
    """CVaR distortion function g(t) = max(0, (t - (1-p))/p) for level p.
    g: [0,1] -> [0,1], concave.
    """
    return max(0.0, (t - (1 - p)) / p)


def distortion_cvar_derivative(t: float, p: float) -> float:
    """Derivative of CVaR distortion: g'(t) = 1/p if t > 1-p, else 0."""
    return 1.0 / p if t > (1 - p) else 0.0


def rho_g_normal(mu: float, sigma: float, g_fn, n_points: int = 1000) -> float:
    """Compute distortion risk rho_g(X) for X ~ N(mu, sigma^2).
    rho_g(X) = integral_0^1 F_X^{-1}(t) g'(t) dt (by the Choquet integral formula).
    """
    t_grid = np.linspace(1e-8, 1 - 1e-8, n_points)
    vals = np.array([stats.norm.ppf(t, mu, sigma) * g_fn(t) for t in t_grid])
    return float(np.trapz(vals, t_grid))


def wasserstein_penalty_constant(g_prime_fn, kappa: float, q: float, integration_points: int = 10000) -> float:
    """Compute the Wasserstein penalty constant from Proposition 6:
    C = (integral_0^1 |1 + kappa * g'(t)|^q dt)^{1/q}
    """
    t_grid = np.linspace(1e-8, 1 - 1e-8, integration_points)
    integrand = np.array([abs(1.0 + kappa * g_prime_fn(t))**q for t in t_grid])
    integral_val = float(np.trapz(integrand, t_grid))
    return integral_val ** (1.0 / q)


def reduced_wasserstein_objective(
    weights: np.ndarray,
    means: np.ndarray,
    cov: np.ndarray,
    kappa: float,
    epsilon: float,
    b_norm: float,  # L^b norm of weights
    wasserstein_const: float,
    p: float  # ES level
) -> float:
    """Compute the Proposition 6 reduced Wasserstein objective (Prop 6 form):
    -rho_{g_kappa}(lambda^T X_0) - epsilon * C * ||lambda||_b
    where rho_{g_kappa} is the distortion risk with modified distortion g_kappa.
    Here we use ES as the distortion risk for concreteness.
    """
    # rho_g_kappa(lambda^T X_0): use normal portfolio distribution
    mu_p = float(np.dot(weights, means))
    sigma_p = float(np.sqrt(weights @ cov @ weights))

    # g_kappa(t) = t + kappa * g(t) applied distortion risk
    # For CVaR distortion, rho_{g_kappa}(X) = mu + sigma * (phi(z)/(1-p)) * (1 + kappa/p) ... simplified
    # Use direct integral for correctness
    def g_kappa_prime(t: float) -> float:
        return 1.0 + kappa * distortion_cvar_derivative(t, p)

    rho_val = rho_g_normal(mu_p, sigma_p, g_kappa_prime)
    penalty = epsilon * wasserstein_const * b_norm
    return -rho_val - penalty


def v_g_star(g_prime_fn, p: float, integration_points: int = 10000) -> float:
    """Compute v_{g^*} for moment uncertainty (Proposition 7):
    v_{g^*} = sqrt(integral_0^1 (g'(t) - g(1))^2 dt)
    For CVaR distortion: g(1) = 1, g'(t) = 1/p for t > 1-p, else 0.
    """
    g1 = 1.0  # g(1) for CVaR distortion
    t_grid = np.linspace(1e-8, 1 - 1e-8, integration_points)
    integrand = np.array([(g_prime_fn(t) - g1)**2 for t in t_grid])
    integral_val = float(np.trapz(integrand, t_grid))
    return float(np.sqrt(integral_val))


def moment_uncertainty_objective(
    weights: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    kappa: float,
    g1: float,
    v_gstar: float
) -> float:
    """Proposition 7 reduced moment-uncertainty objective:
    -(1 + kappa*g(1)) * lambda^T mu - kappa * v_{g^*} * sqrt(lambda^T Sigma lambda)
    (maximization form; we minimize the negative)
    """
    mean_term = (1 + kappa * g1) * float(np.dot(weights, mu))
    sd_term = kappa * v_gstar * float(np.sqrt(weights @ Sigma @ weights))
    return -(mean_term + sd_term)  # negative because we want to maximize


def validate_proposition6(p: float = 0.95):
    """Validate Proposition 6: Wasserstein reduction for ES distortion."""
    print('\n--- Proposition 6 (Wasserstein Uncertainty) Validation ---')
    n = 3
    mu = np.array([4.0, 4.8, 5.3])
    Sigma = np.diag([6.76, 0.81, 0.04])
    kappa = 0.5
    epsilon = 0.1
    q = 2.0  # Holder conjugate of b=2
    b = 2.0

    g_prime = lambda t: distortion_cvar_derivative(t, p)
    C = wasserstein_penalty_constant(g_prime, kappa, q)
    print(f'  ES level p={p}, kappa={kappa}, epsilon={epsilon}')
    print(f'  Wasserstein penalty constant C = {C:.6f}')

    # Evaluate at 3 simplex vertices
    vertices = [np.eye(n)[i] for i in range(n)]
    records = []
    for i, w in enumerate(vertices):
        b_norm = float(np.linalg.norm(w, b))
        obj = reduced_wasserstein_objective(w, mu, Sigma, kappa, epsilon, b_norm, C, p)
        records.append({'Vertex': f'e_{i+1}', 'Reduced_Obj': round(obj, 6), 'b_norm': round(b_norm, 6)})

    df = pd.DataFrame(records)
    print_table(df, 'Prop 6: Wasserstein Reduced Objective at Vertices')
    return df


def validate_proposition7(p: float = 0.95):
    """Validate Proposition 7: moment uncertainty reduction for ES distortion."""
    print('\n--- Proposition 7 (Moment Uncertainty) Validation ---')
    n = 3
    mu = np.array([4.9, 4.1, 5.5])
    # Section 8 covariance
    sigma_vec = np.array([1.8, 0.2, 1.0])
    rho12, rho13, rho23 = 0.30, 0.20, 0.25
    Sigma = np.array([
        [1.8**2, rho12*1.8*0.2, rho13*1.8*1.0],
        [rho12*1.8*0.2, 0.2**2, rho23*0.2*1.0],
        [rho13*1.8*1.0, rho23*0.2*1.0, 1.0**2]
    ])

    g_prime = lambda t: distortion_cvar_derivative(t, p)
    g1 = 1.0  # g(1) = 1 for CVaR distortion
    vg = v_g_star(g_prime, p)
    print(f'  ES level p={p}')
    print(f'  v_g^* = {vg:.6f}')

    kappa_values = [0.5, 1.0, 2.0]
    vertices = [np.eye(n)[i] for i in range(n)]
    records = []
    for kappa in kappa_values:
        for i, w in enumerate(vertices):
            obj = moment_uncertainty_objective(w, mu, Sigma, kappa, g1, vg)
            records.append({
                'kappa': kappa, 'Vertex': f'e_{i+1}',
                'Moment_Obj': round(obj, 6)
            })

    df = pd.DataFrame(records)
    print_table(df, 'Prop 7: Moment Uncertainty Reduced Objective at Vertices')

    # Validate special case: optimal vertex for large kappa
    print('\n  Marginal ES_p values (mu_i + sigma_i * c_p):')
    z = stats.norm.ppf(p)
    c_p = stats.norm.pdf(z) / (1 - p)
    for i in range(n):
        es_i = mu[i] + sigma_vec[i] * c_p
        print(f'    Asset {i+1}: {es_i:.4f}')
    print(f'  Minimum marginal ES_p: Asset {np.argmin(mu + sigma_vec*c_p)+1}')

    return df


def main():
    print('\n' + '='*60)
    print('EXPERIMENT 7: Structured Uncertainty Reductions (Propositions 6 & 7)')
    print('='*60)

    df_p6 = validate_proposition6(p=0.95)
    df_p7 = validate_proposition7(p=0.95)

    # Save
    df_p6.to_csv(RESULTS_DIR / 'exp7_prop6.csv', index=False)
    df_p7.to_csv(RESULTS_DIR / 'exp7_prop7.csv', index=False)

    # Simple visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Exp 7: Structured Uncertainty Reductions', fontsize=13)

    ax = axes[0]
    ax.bar(df_p6['Vertex'], -df_p6['Reduced_Obj'], color=['steelblue', 'coral', 'green'])
    ax.set_title('Prop 6: -Reduced Wasserstein Obj at Vertices')
    ax.set_ylabel('Objective Value')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for kappa in [0.5, 1.0, 2.0]:
        subset = df_p7[df_p7['kappa'] == kappa]
        ax.bar([f'{v}_{kappa}' for v in subset['Vertex']], -subset['Moment_Obj'],
               label=f'kappa={kappa}', alpha=0.7)
    ax.set_title('Prop 7: -Moment Uncertainty Obj at Vertices')
    ax.set_ylabel('Objective Value')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp7_propositions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {RESULTS_DIR}/exp7_propositions.png')

    return df_p6, df_p7


if __name__ == '__main__':
    main()
