import numpy as np
import pytest
from scipy import stats
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk_measures import (
    normal_var, normal_es, normal_rvar,
    lognormal_var, lognormal_rvar,
    portfolio_sd, portfolio_normal_var, portfolio_normal_es
)


# ---------------------------------------------------------------------------
# normal_var
# ---------------------------------------------------------------------------
def test_normal_var_standard():
    """VaR of standard normal at p=0.99."""
    result = normal_var(0.0, 1.0, 0.99)
    expected = stats.norm.ppf(0.99)
    assert abs(result - expected) < 1e-10


def test_normal_var_asset1_table2():
    """Table 2 check: VaR_0.99^+(X1) = 10.0485."""
    result = normal_var(4.0, 2.6, 0.99)
    assert abs(result - 10.0485) < 5e-4, f'Got {result}'


def test_normal_var_asset2_table2():
    """Table 2 check: VaR_0.99^+(X2) = 6.8937."""
    result = normal_var(4.8, 0.9, 0.99)
    assert abs(result - 6.8937) < 5e-4, f'Got {result}'


def test_normal_var_monotone_in_p():
    """VaR is monotone in p."""
    vars = [normal_var(0, 1, p) for p in [0.90, 0.95, 0.99]]
    assert vars[0] < vars[1] < vars[2]


def test_normal_var_positive_sd_shift():
    """Larger sigma increases VaR."""
    assert normal_var(0, 2, 0.99) > normal_var(0, 1, 0.99)


# ---------------------------------------------------------------------------
# normal_es
# ---------------------------------------------------------------------------
def test_normal_es_exceeds_var():
    """ES_p >= VaR_p for any distribution."""
    mu, sigma, p = 5.0, 2.0, 0.95
    assert normal_es(mu, sigma, p) >= normal_var(mu, sigma, p) - 1e-10


def test_normal_es_standard():
    """ES of N(0,1) at p=0.95: approximately 2.0627."""
    result = normal_es(0.0, 1.0, 0.95)
    assert abs(result - 2.0627) < 1e-3


# ---------------------------------------------------------------------------
# normal_rvar
# ---------------------------------------------------------------------------
def test_normal_rvar_asset1_table2():
    """Table 2: R_{0.01,0.10}(X1) = 8.1960."""
    result = normal_rvar(4.0, 2.6, 0.01, 0.10)
    assert abs(result - 8.1960) < 5e-3, f'Got {result}'


def test_normal_rvar_asset2_table2():
    """Table 2: R_{0.01,0.10}(X2) = 6.2525."""
    result = normal_rvar(4.8, 0.9, 0.01, 0.10)
    assert abs(result - 6.2525) < 5e-3, f'Got {result}'


def test_rvar_bounds():
    """RVaR should be between VaR_{1-beta-alpha} and VaR_{1-beta}."""
    mu, sigma = 4.0, 2.0
    beta, alpha = 0.01, 0.10
    rvar = normal_rvar(mu, sigma, beta, alpha)
    var_lo = normal_var(mu, sigma, 1 - beta - alpha)
    var_hi = normal_var(mu, sigma, 1 - beta)
    assert var_lo <= rvar + 1e-8
    assert rvar <= var_hi + 1e-8


# ---------------------------------------------------------------------------
# lognormal_var
# ---------------------------------------------------------------------------
def test_lognormal_var_asset3_table2():
    """Table 2: VaR_0.99^+(X3) = 6.0951."""
    from src.marginals import asset3_params
    a, b = asset3_params(eta=1.0)
    result = lognormal_var(a, b, 0.99)
    assert abs(result - 6.0951) < 5e-3, f'Got {result}'


def test_lognormal_rvar_asset3_table2():
    """Table 2: RVaR_0.01_0.10(X3) = 5.6327."""
    from src.marginals import asset3_params
    a, b = asset3_params(eta=1.0)
    result = lognormal_rvar(a, b, 0.01, 0.10)
    assert abs(result - 5.6327) < 5e-3, f'Got {result}'


# ---------------------------------------------------------------------------
# portfolio_sd
# ---------------------------------------------------------------------------
def test_portfolio_sd_concentrated():
    """Concentrated portfolio e_i has SD = sigma_i."""
    Sigma = np.diag([6.76, 0.81, 0.04])
    w1 = np.array([1., 0., 0.])
    assert abs(portfolio_sd(w1, Sigma) - 2.6) < 1e-10


def test_portfolio_sd_uniform_exchangeable():
    """For iid sigma^2*I, SD(uniform weights) = sigma/sqrt(n)."""
    n = 3
    sigma = 2.0
    Sigma = sigma**2 * np.eye(n)
    w = np.ones(n) / n
    expected = sigma / np.sqrt(n)
    assert abs(portfolio_sd(w, Sigma) - expected) < 1e-10


def test_portfolio_sd_nonnegative():
    """SD is always non-negative."""
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    w = np.array([0.4, 0.6])
    assert portfolio_sd(w, Sigma) >= 0
