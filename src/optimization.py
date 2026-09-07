import numpy as np
import cvxpy as cp
from typing import Tuple, Optional


def _psd_factor(cov: np.ndarray) -> np.ndarray:
    """Return L such that L @ L.T = cov for a positive semi-definite matrix.
    Uses eigenvalue decomposition to handle rank-deficient (PSD but not PD) inputs.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals))


def solve_mean_sd(
    means: np.ndarray,
    cov: np.ndarray,
    kappa: float,
    n_restarts: int = 10
) -> Tuple[np.ndarray, float]:
    """Solve min_{lambda in Delta_n} lambda^T m + kappa * sqrt(lambda^T Sigma lambda).
    Returns (optimal_weights, optimal_objective_value).
    Uses cvxpy with ECOS solver. Handles PSD (rank-deficient) covariance matrices.
    """
    n = len(means)
    lam = cp.Variable(n, nonneg=True)
    L = _psd_factor(cov)
    obj = means @ lam + kappa * cp.norm(L.T @ lam, 2)
    constraints = [cp.sum(lam) == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-9, tol_gap_rel=1e-9, max_iter=10000)
    if lam.value is None:
        raise ValueError(f'Optimization failed: status={prob.status}')
    w = np.maximum(lam.value, 0)
    w /= w.sum()
    return w, float(prob.value)


def solve_mean_sd_blended(
    means: np.ndarray,
    sigmas: np.ndarray,  # marginal SDs
    cov: np.ndarray,
    kappa: float,
    c_p: float,
    omega: float
) -> Tuple[np.ndarray, float]:
    """Solve blended weighted-robustness objective (Section 8):
    min_{lambda in Delta_n} lambda^T mu + c_p * [omega * lambda^T sigma + (1-omega) * sqrt(lambda^T Sigma lambda)]
    Handles PSD (rank-deficient) covariance matrices.
    """
    n = len(means)
    lam = cp.Variable(n, nonneg=True)
    L = _psd_factor(cov)
    ref_sd = cp.norm(L.T @ lam, 2)
    wc_term = sigmas @ lam
    obj = means @ lam + c_p * (omega * wc_term + (1 - omega) * ref_sd)
    constraints = [cp.sum(lam) == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-9, tol_gap_rel=1e-9, max_iter=10000)
    if lam.value is None:
        raise ValueError(f'Blended optimization failed: status={prob.status}')
    w = np.maximum(lam.value, 0)
    w /= w.sum()
    return w, float(prob.value)


def herfindahl_hirschman(weights: np.ndarray) -> float:
    """HHI = sum_i lambda_i^2. Measures portfolio concentration (1 = fully concentrated)."""
    return float(np.sum(np.asarray(weights)**2))
