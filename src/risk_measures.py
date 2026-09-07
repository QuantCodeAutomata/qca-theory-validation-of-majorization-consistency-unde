import numpy as np
from scipy import stats, integrate
from typing import Union, Tuple


def normal_var(mu: float, sigma: float, p: float) -> float:
    """VaR_p^+(X) for X ~ N(mu, sigma^2). For continuous distributions VaR^+ = VaR."""
    return mu + sigma * stats.norm.ppf(p)


def normal_es(mu: float, sigma: float, p: float) -> float:
    """Expected Shortfall ES_p(X) for X ~ N(mu, sigma^2).
    ES_p = mu + sigma * phi(Phi^{-1}(p)) / (1 - p)
    """
    z = stats.norm.ppf(p)
    return mu + sigma * stats.norm.pdf(z) / (1 - p)


def normal_rvar(mu: float, sigma: float, beta: float, alpha: float) -> float:
    """RVaR_{beta,alpha}(X) = (1/alpha) * integral_{1-beta-alpha}^{1-beta} F_X^{-1}(t) dt
    For X ~ N(mu, sigma^2). Computed via numerical integration.
    beta: lower tail parameter (e.g. 0.01)
    alpha: integration width (e.g. 0.10)
    """
    def quantile_fn(t: float) -> float:
        return mu + sigma * stats.norm.ppf(t)
    result, _ = integrate.quad(quantile_fn, 1 - beta - alpha, 1 - beta, limit=100)
    return result / alpha


def lognormal_var(a: float, b: float, p: float) -> float:
    """VaR_p^+(X) for X = a + b*exp(Z), Z ~ N(0,1).
    F_X^{-1}(p) = a + b * exp(Phi^{-1}(p))
    """
    return a + b * np.exp(stats.norm.ppf(p))


def lognormal_rvar(a: float, b: float, beta: float, alpha: float) -> float:
    """RVaR_{beta,alpha}(X) for X = a + b*exp(Z), Z ~ N(0,1)."""
    def quantile_fn(t: float) -> float:
        return a + b * np.exp(stats.norm.ppf(t))
    result, _ = integrate.quad(quantile_fn, 1 - beta - alpha, 1 - beta, limit=200)
    return result / alpha


def portfolio_sd(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Standard deviation of portfolio return: sqrt(lambda^T Sigma lambda)."""
    w = np.asarray(weights)
    return float(np.sqrt(w @ cov_matrix @ w))


def portfolio_normal_var(weights: np.ndarray, means: np.ndarray, cov_matrix: np.ndarray, p: float) -> float:
    """VaR_p^+(lambda^T X) for multivariate normal X."""
    mu_p = float(np.dot(weights, means))
    sigma_p = portfolio_sd(weights, cov_matrix)
    return normal_var(mu_p, sigma_p, p)


def portfolio_normal_es(weights: np.ndarray, means: np.ndarray, cov_matrix: np.ndarray, p: float) -> float:
    """ES_p(lambda^T X) for multivariate normal X."""
    mu_p = float(np.dot(weights, means))
    sigma_p = portfolio_sd(weights, cov_matrix)
    return normal_es(mu_p, sigma_p, p)
