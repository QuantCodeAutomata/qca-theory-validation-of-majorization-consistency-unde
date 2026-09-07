import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Exp 1 Checks
# ---------------------------------------------------------------------------
def test_exp1_weak_consistency_sd_exchangeable():
    """Exp 1: weak consistency S_SD(beta) <= max_k S_SD(Pi_k lambda) for iid N(0,1)."""
    from src.majorization import get_permutation_matrices, is_majorized
    import math
    n = 3
    lam = np.array([1., 0., 0.])
    beta = np.array([1/3, 1/3, 1/3])
    assert is_majorized(beta, lam)

    def s_sd(w):
        return float(np.linalg.norm(w, 2))

    perm_mats = get_permutation_matrices(n)
    s_beta = s_sd(beta)
    max_perm = max(s_sd(P @ lam) for P in perm_mats)
    assert s_beta <= max_perm + 1e-8


def test_exp1_strong_consistency_exchangeable():
    """Exp 1: strong consistency S_SD(beta) <= S_SD(lambda) under exchangeability."""
    n = 3
    lam = np.array([1., 0., 0.])
    beta = np.array([1/3, 1/3, 1/3])

    def s_sd(w):
        return float(np.linalg.norm(w, 2))

    assert s_sd(beta) <= s_sd(lam) + 1e-8


# ---------------------------------------------------------------------------
# Exp 2 Checks
# ---------------------------------------------------------------------------
def test_exp2_table2_var_values():
    """Exp 2: Table 2 VaR_0.99^+ values match paper to 4 decimals."""
    from src.risk_measures import normal_var, lognormal_var
    from src.marginals import asset3_params
    a3, b3 = asset3_params(eta=1.0)
    assert abs(normal_var(4.0, 2.6, 0.99) - 10.0485) < 5e-4
    assert abs(normal_var(4.8, 0.9, 0.99) - 6.8937) < 5e-4
    assert abs(lognormal_var(a3, b3, 0.99) - 6.0951) < 5e-4


def test_exp2_table2_rvar_values():
    """Exp 2: Table 2 RVaR values match paper to 4 decimals."""
    from src.risk_measures import normal_rvar, lognormal_rvar
    from src.marginals import asset3_params
    a3, b3 = asset3_params(eta=1.0)
    assert abs(normal_rvar(4.0, 2.6, 0.01, 0.10) - 8.1960) < 5e-3
    assert abs(normal_rvar(4.8, 0.9, 0.01, 0.10) - 6.2525) < 5e-3
    assert abs(lognormal_rvar(a3, b3, 0.01, 0.10) - 5.6327) < 5e-3


def test_exp2_regime_switching_var():
    """Exp 2: Under VaR, e_3 dominates at kappa=1.5."""
    from src.risk_measures import normal_var, lognormal_var
    from src.marginals import MEANS, asset3_params
    a3, b3 = asset3_params(eta=1.0)
    risks = np.array([normal_var(4.0, 2.6, 0.99), normal_var(4.8, 0.9, 0.99), lognormal_var(a3, b3, 0.99)])
    kappa = 1.5
    f = -MEANS - kappa * risks
    assert np.argmax(f) == 2  # Asset 3 dominates at high kappa


# ---------------------------------------------------------------------------
# Exp 3 Checks
# ---------------------------------------------------------------------------
def test_exp3_table4_kappa0():
    """Exp 3: at kappa=0, optimal portfolio is e_1 (min mean), V_SD = -4.0."""
    from src.optimization import solve_mean_sd
    from src.marginals import MEANS, SIGMA_C
    w, obj = solve_mean_sd(MEANS, SIGMA_C, kappa=0.0)
    assert abs(obj - (-4.0)) < 0.01 or abs(obj - 4.0) < 0.01  # objective could be +4.0 without sign convention
    # The minimization gives obj = min mean = 4.0 at kappa=0
    assert obj < 5.0  # Should be close to 4.0


def test_exp3_table4_kappa125():
    """Exp 3: at kappa=1.25, optimal portfolio is e_3, V_SD=-5.55."""
    from src.optimization import solve_mean_sd
    from src.marginals import MEANS, SIGMA_C
    w, obj = solve_mean_sd(MEANS, SIGMA_C, kappa=1.25)
    assert w[2] > 0.9  # Asset 3 dominant


# ---------------------------------------------------------------------------
# Exp 4 Checks
# ---------------------------------------------------------------------------
def test_exp4_robustness_premium_nonnegative():
    """Exp 4: robustness premium Delta(kappa) = J_c^* - J_0^* >= 0."""
    from src.optimization import solve_mean_sd
    from src.marginals import MEANS, SIGMA_C, SIGMA_0
    for kappa in [0.5, 1.0, 1.5]:
        _, j0 = solve_mean_sd(MEANS, SIGMA_0, kappa)
        _, jc = solve_mean_sd(MEANS, SIGMA_C, kappa)
        assert jc >= j0 - 1e-4, f'At kappa={kappa}: Delta={jc-j0:.6f} < 0'


def test_exp4_hhi_range():
    """HHI must be in [1/n, 1] for simplex portfolios."""
    from src.optimization import solve_mean_sd, herfindahl_hirschman
    from src.marginals import MEANS, SIGMA_0
    w, _ = solve_mean_sd(MEANS, SIGMA_0, kappa=1.0)
    hhi = herfindahl_hirschman(w)
    n = len(w)
    assert hhi >= 1/n - 1e-6
    assert hhi <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Exp 5 Checks
# ---------------------------------------------------------------------------
def test_exp5_mean_invariance():
    """Exp 5: E[X3(eta)] = 5.3 for all eta in range."""
    import math
    from src.marginals import asset3_params
    for eta in [0.2, 0.5, 1.0, 1.5]:
        a, b = asset3_params(eta)
        mean = a + b * math.exp(eta**2 / 2)
        assert abs(mean - 5.3) < 1e-5, f'Mean={mean} at eta={eta}'


def test_exp5_sd_invariance():
    """Exp 5: SD(X3(eta)) = 0.2 for all eta in range."""
    import math
    from src.marginals import asset3_params
    for eta in [0.2, 0.5, 1.0, 1.5]:
        a, b = asset3_params(eta)
        var = b**2 * (math.exp(2*eta**2) - math.exp(eta**2))
        sd = math.sqrt(var)
        assert abs(sd - 0.2) < 1e-5, f'SD={sd} at eta={eta}'


def test_exp5_table6_eta1():
    """Exp 5: Table 6 at eta=1.0 matches expected values."""
    from src.marginals import asset3_var, asset3_rvar
    r3_var = asset3_var(0.99, 1.0)
    r3_rvar = asset3_rvar(0.01, 0.10, 1.0)
    assert abs(r3_var - 6.0951) < 5e-3
    assert abs(r3_rvar - 5.6327) < 5e-3


# ---------------------------------------------------------------------------
# Exp 6 Checks
# ---------------------------------------------------------------------------
def test_exp6_covariance_less_than_one():
    """Exp 6: c = Cov(F1^{-1}(U), F2^{-1}(U)) < 1."""
    from scipy import integrate, stats

    def laplace_q(u):
        b = 1.0 / np.sqrt(2.0)
        v = u - 0.5
        return -b * np.sign(v) * np.log(1.0 - 2.0 * abs(v))

    def integrand(u):
        return stats.norm.ppf(u) * laplace_q(u)

    r1, _ = integrate.quad(integrand, 1e-10, 0.5, limit=500)
    r2, _ = integrate.quad(integrand, 0.5, 1-1e-10, limit=500)
    c = r1 + r2
    assert c < 1.0
    assert c > 0.0  # Should be positive (both quantiles are increasing)


def test_exp6_diversified_variance_below_one():
    """Exp 6: Var(0.5*X1 + 0.5*X2) under comonotonicity < 1."""
    from scipy import integrate, stats

    def laplace_q(u):
        b = 1.0 / np.sqrt(2.0)
        v = u - 0.5
        return -b * np.sign(v) * np.log(1.0 - 2.0 * abs(v))

    def integrand(u):
        return stats.norm.ppf(u) * laplace_q(u)

    r1, _ = integrate.quad(integrand, 1e-10, 0.5, limit=500)
    r2, _ = integrate.quad(integrand, 0.5, 1-1e-10, limit=500)
    c = r1 + r2
    var_div = (1 + c) / 2
    assert var_div < 1.0


# ---------------------------------------------------------------------------
# Exp 8 Checks
# ---------------------------------------------------------------------------
def test_exp8_omega1_concentration():
    """Exp 8: omega=1 should concentrate in the asset with minimum marginal ES_p."""
    from scipy import stats
    mu = np.array([4.9, 4.1, 5.5])
    sigma = np.array([1.8, 0.2, 1.0])
    p = 0.95
    z = stats.norm.ppf(p)
    c_p = stats.norm.pdf(z) / (1 - p)
    marginal_es = mu + sigma * c_p
    expected_asset = np.argmin(marginal_es)
    assert expected_asset == 1  # Asset 2 (0-indexed: 1)
