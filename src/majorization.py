import numpy as np
from itertools import permutations
from typing import List, Tuple


def is_majorized(beta: np.ndarray, lam: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if beta is majorized by lam (beta << lam).
    beta << lam iff for all k: sum of k largest components of beta <= sum of k largest of lam,
    with equality for k = n.
    """
    beta_sorted = np.sort(beta)[::-1]
    lam_sorted = np.sort(lam)[::-1]
    n = len(beta)
    for k in range(1, n + 1):
        if np.sum(beta_sorted[:k]) > np.sum(lam_sorted[:k]) + tol:
            return False
    return True


def get_permutation_matrices(n: int) -> List[np.ndarray]:
    """Return all n! permutation matrices for dimension n."""
    perm_mats = []
    for perm in permutations(range(n)):
        P = np.zeros((n, n))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        perm_mats.append(P)
    return perm_mats


def apply_permutation(P: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Apply permutation matrix P to weight vector lam: returns P @ lam."""
    return P @ lam


def construct_beta_from_permutations(lam: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Construct beta = sum_k w_k * Pi_k @ lam (Birkhoff decomposition).
    lam: portfolio weight vector
    weights: convex combination weights (sum to 1, all >= 0)
    Returns beta and list of permuted vectors Pi_k @ lam.
    """
    n = len(lam)
    perm_mats = get_permutation_matrices(n)
    assert len(weights) == len(perm_mats), f'Need {len(perm_mats)} weights for n={n}'
    assert abs(sum(weights) - 1.0) < 1e-8
    permuted = [P @ lam for P in perm_mats]
    beta = sum(w * v for w, v in zip(weights, permuted))
    return beta, permuted


def uniform_mixture_beta(lam: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Construct beta as uniform mixture over all permutations of lam.
    This yields the most diversified (uniform) portfolio if lam is not already uniform.
    """
    n = len(lam)
    from math import factorial
    n_perms = factorial(n)
    weights = [1.0 / n_perms] * n_perms
    return construct_beta_from_permutations(lam, weights)


def verify_doubly_stochastic(D: np.ndarray, tol: float = 1e-6) -> bool:
    """Verify that matrix D is doubly stochastic: rows and cols sum to 1, all entries >= 0."""
    if np.any(D < -tol):
        return False
    if not np.allclose(D.sum(axis=1), 1.0, atol=tol):
        return False
    if not np.allclose(D.sum(axis=0), 1.0, atol=tol):
        return False
    return True
