# Repository: Portfolio Diversification Theory Validation

## Purpose
Numerical validation of theoretical results on portfolio diversification under majorization order and dependence ambiguity.

## Key Concepts
- Majorization order: beta << lambda means beta is more diversified
- Weak consistency: rho is weakly consistent with majorization iff S_rho(beta) <= max_k S_rho(Pi_k lambda)
- RVaR_{beta,alpha}(X) = (1/alpha) integral_{1-beta-alpha}^{1-beta} F_X^{-1}(t) dt
- Comonotonic covariance: Cov(X_i, X_j) = Cov(F_i^{-1}(U), F_j^{-1}(U))

## Section 7 Asset Marginals
- X1 = 4 + 2.6*Z1, Z1 ~ N(0,1): mu=4.0, sigma=2.6
- X2 = 4.8 + 0.9*Z2, Z2 ~ N(0,1): mu=4.8, sigma=0.9
- X3 = 5.147425 + 0.092541*exp(Z3), Z3 ~ N(0,1): mu=5.3, sigma=0.2

## Expected Table 2 Values
- Asset 1: mean=4.0000, sd=2.6000, VaR_0.99=10.0485, RVaR=8.1960
- Asset 2: mean=4.8000, sd=0.9000, VaR_0.99=6.8937, RVaR=6.2525
- Asset 3: mean=5.3000, sd=0.2000, VaR_0.99=6.0951, RVaR=5.6327

## Key Packages
- cvxpy: portfolio optimization (solver: CLARABEL — ECOS is not installed in this environment)
- scipy.integrate.quad: RVaR integration
- scipy.stats.norm: normal quantiles
- numpy: linear algebra, permutations

## Important Implementation Notes
- SIGMA_C (comonotonic covariance) is rank-2, not positive definite: X1 and X2 are both
  linear in the same standard normal Z, so their rows are proportional. Cholesky fails.
  Use `_psd_factor(cov)` (eigenvalue-based square root) in optimization.py instead.
- CLARABEL solver options: `tol_gap_abs`, `tol_gap_rel`, `max_iter` (not `eps_abs`/`eps_rel`).
- `_psd_factor` returns L such that L @ L.T = cov for any PSD matrix.

## Experiment Notes
- exp4_dependence_info.py: Table 5 + Figure 2(a)-(b). Compares SIGMA_0 vs SIGMA_C optimization over kappa. Runs 2*200 CLARABEL solves for the kappa-grid paths; robustness premium delta=jc-j0 is always >= 0.
- exp5_tail_shape.py: Table 6 + Figure 2(c)-(d). Closed-form switching thresholds kappa*(eta) = (m_hi-m_lo)/(r_lo-r_hi). Phase maps computed analytically (no solver). asset3_params(eta) preserves mean=5.3, sd=0.2 for all eta.
- exp6_counterexample.py: Normal N(0,1) vs Laplace(0,1/√2) under comonotonic coupling. c = Cov(Φ^{-1}(U), F2^{-1}(U)) ≈ 0.9813 < 1, so Var(0.5,0.5) = (1+c)/2 < 1. IntegrationWarning from quad near the Laplace kink at u=0.5 is expected; result still passes all assertions.

## Experiment Files
- exp1–exp8 are in exp/; run individually or via `python exp/run_all.py`
- exp7: Propositions 6 & 7 (Wasserstein + moment uncertainty reductions); uses CVaR distortion with Section 8 Gaussian inputs
- exp8: Blended ES objective with omega ∈ [0,1]; omega=1 → LP → concentration in Asset 2 (min marginal ES_p)
- `print_table(df, title)` from src/utils accepts both dict and DataFrame

## Testing
All tests are in tests/ and use pytest. Run with: pytest tests/ -v
- tests/test_risk_measures.py: 15 unit tests for risk measure functions
- tests/test_majorization.py: 13 unit tests for majorization utilities
- tests/test_experiments.py: 15 integration tests validating key experiment outputs
