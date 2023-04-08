import pytest
import numpy as np
from exp_decay import ExponentialDecay


@pytest.mark.parametrize("u, t, a, expected", [(3.2, 1, 0.4, -1.28)])
def test_dudt(u, t, a, expected):
    """Tests call method."""
    verify = ExponentialDecay(a)
    calculated = verify(t, u)
    tol = 10**-5
    assert abs(calculated - expected) < tol


def test_negative_a():
    """Tests if __init__ raises ValueError."""
    with pytest.raises(ValueError):
        ExponentialDecay(-1)


@pytest.mark.parametrize("alpha, u0, T, dt",
                        [(0.6, 3, 6, 0.01),
                         (0.2, 20 , 30, 0.0001)])
def test_solve(alpha, u0, T, dt):
    """Tests if solve method calculates correct values."""
    decay_model = ExponentialDecay(alpha)
    t, u_solved = np.array(decay_model.solve(u0, T, dt))
    tol = 10**-2
    u_func = lambda t: u0 * np.exp(-alpha*t)
    u_exp = u_func(t)
    error_str = "Calculated solution not close enough to expected solution."
    assert np.all(abs(u_exp - u_solved) < tol), error_str

