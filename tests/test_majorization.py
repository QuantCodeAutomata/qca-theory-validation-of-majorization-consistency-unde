import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.majorization import (
    is_majorized, get_permutation_matrices,
    construct_beta_from_permutations, uniform_mixture_beta,
    verify_doubly_stochastic
)


def test_is_majorized_uniform_by_concentrated():
    """Uniform (1/3,1/3,1/3) is majorized by (1,0,0)."""
    assert is_majorized(np.array([1/3, 1/3, 1/3]), np.array([1., 0., 0.]))


def test_is_majorized_self():
    """Every vector majorizes itself."""
    lam = np.array([0.5, 0.3, 0.2])
    assert is_majorized(lam, lam)


def test_is_majorized_asymmetric():
    """(0.7,0.2,0.1) majorizes (0.5,0.3,0.2) but not vice versa."""
    a = np.array([0.7, 0.2, 0.1])
    b = np.array([0.5, 0.3, 0.2])
    assert is_majorized(b, a)
    assert not is_majorized(a, b)


def test_is_majorized_requires_equal_sum():
    """Majorization requires equal sums (both are simplex elements here)."""
    a = np.array([0.8, 0.1, 0.1])
    b = np.array([0.4, 0.4, 0.2])
    assert is_majorized(b, a)


def test_get_permutation_matrices_n2():
    """For n=2, there are 2 permutation matrices."""
    mats = get_permutation_matrices(2)
    assert len(mats) == 2


def test_get_permutation_matrices_n3():
    """For n=3, there are 6 permutation matrices."""
    mats = get_permutation_matrices(3)
    assert len(mats) == 6


def test_permutation_matrices_are_doubly_stochastic():
    """Each permutation matrix is doubly stochastic."""
    for P in get_permutation_matrices(3):
        assert verify_doubly_stochastic(P)


def test_uniform_mixture_is_majorized():
    """Uniform mixture of permutations of lambda is majorized by lambda."""
    lam = np.array([0.7, 0.2, 0.1])
    beta, _ = uniform_mixture_beta(lam)
    assert is_majorized(beta, lam)


def test_uniform_mixture_is_uniform_for_n3():
    """Uniform mixture of all permutations of any simplex element should give (1/3, 1/3, 1/3)."""
    lam = np.array([1., 0., 0.])
    beta, _ = uniform_mixture_beta(lam)
    expected = np.array([1/3, 1/3, 1/3])
    np.testing.assert_allclose(beta, expected, atol=1e-10)


def test_construct_beta_sums_to_one():
    """Beta constructed from permutations sums to 1."""
    from math import factorial
    n = 3
    lam = np.array([0.6, 0.3, 0.1])
    n_perms = factorial(n)
    weights = np.ones(n_perms) / n_perms
    beta, _ = construct_beta_from_permutations(lam, weights)
    assert abs(beta.sum() - 1.0) < 1e-10


def test_verify_doubly_stochastic_identity():
    """Identity matrix is doubly stochastic."""
    assert verify_doubly_stochastic(np.eye(3))


def test_verify_doubly_stochastic_uniform():
    """Uniform (1/n)*ones matrix is doubly stochastic."""
    n = 3
    D = np.ones((n, n)) / n
    assert verify_doubly_stochastic(D)


def test_verify_doubly_stochastic_negative_fails():
    """Matrix with negative entries is not doubly stochastic."""
    D = np.array([[1.1, -0.1], [-0.1, 1.1]])
    assert not verify_doubly_stochastic(D)
