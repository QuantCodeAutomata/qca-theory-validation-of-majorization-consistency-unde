import numpy as np
from scipy import stats, integrate
from typing import Dict, Tuple
import math

# Section 7 base asset parameters
ASSET_PARAMS = {
    'X1': {'type': 'normal', 'mu': 4.0, 'sigma': 2.6},
    'X2': {'type': 'normal', 'mu': 4.8, 'sigma': 0.9},
    'X3': {'type': 'lognormal', 'a': 5.147425, 'b': 0.092541},
}

MEANS = np.array([4.0, 4.8, 5.3])
SDS = np.array([2.6, 0.9, 0.2])

# Worst-case comonotonic covariance matrix (Section 7.2)
SIGMA_C = np.array([
    [6.7600, 2.3400, 0.3967],
    [2.3400, 0.8100, 0.1373],
    [0.3967, 0.1373, 0.0400]
])

# Independence covariance matrix
SIGMA_0 = np.diag([6.76, 0.81, 0.04])


def asset3_params(eta: float = 1.0) -> Tuple[float, float]:
    """Compute a(eta) and b(eta) for X3(eta) = a(eta) + b(eta)*exp(eta*Z).
    Preserves E[X3] = 5.3 and SD(X3) = 0.2 for all eta.
    b(eta) = 0.2 / sqrt((exp(eta^2)-1) * exp(eta^2))
    a(eta) = 5.3 - b(eta) * exp(eta^2 / 2)
    """
    b = 0.2 / math.sqrt((math.exp(eta**2) - 1) * math.exp(eta**2))
    a = 5.3 - b * math.exp(eta**2 / 2)
    return a, b


def asset3_quantile(t: float, eta: float = 1.0) -> float:
    """Quantile function F_{X3}^{-1}(t) for X3(eta) = a + b*exp(eta*Z)."""
    a, b = asset3_params(eta)
    return a + b * np.exp(eta * stats.norm.ppf(t))


def asset3_var(p: float = 0.99, eta: float = 1.0) -> float:
    """VaR_p^+(X3(eta)) = a(eta) + b(eta) * exp(eta * Phi^{-1}(p))."""
    a, b = asset3_params(eta)
    return a + b * np.exp(eta * stats.norm.ppf(p))


def asset3_rvar(beta: float = 0.01, alpha: float = 0.10, eta: float = 1.0) -> float:
    """RVaR_{beta,alpha}(X3(eta)) via numerical quadrature."""
    a, b = asset3_params(eta)
    def quantile_fn(t):
        return a + b * np.exp(eta * stats.norm.ppf(t))
    lo = 1 - beta - alpha
    hi = 1 - beta
    result, _ = integrate.quad(quantile_fn, lo, hi, limit=200)
    return result / alpha


def compute_table2(var_p: float = 0.99, rvar_beta: float = 0.01, rvar_alpha: float = 0.10) -> Dict:
    """Compute Table 2: asset statistics for the 3 Section 7 marginals."""
    from src.risk_measures import normal_var, normal_rvar, lognormal_var, lognormal_rvar
    a3, b3 = asset3_params(eta=1.0)
    result = {
        'Asset': [1, 2, 3],
        'Mean': [4.0, 4.8, 5.3],
        'SD': [2.6, 0.9, 0.2],
        'VaR_0.99': [
            normal_var(4.0, 2.6, var_p),
            normal_var(4.8, 0.9, var_p),
            lognormal_var(a3, b3, var_p)
        ],
        'RVaR_0.01_0.10': [
            normal_rvar(4.0, 2.6, rvar_beta, rvar_alpha),
            normal_rvar(4.8, 0.9, rvar_beta, rvar_alpha),
            lognormal_rvar(a3, b3, rvar_beta, rvar_alpha)
        ]
    }
    return result
