import pytest
import numpy as np
from pendulum import Pendulum


@pytest.mark.parametrize("L, M, theta, omega, expected",
                        [(2.7, 1, np.pi/6, 0.15, (0.15,-1.816667)),
                         (2.7, 1, 0, 0, (0,0))])
def test_pendulum_call(L, M, theta, omega, expected):
    """Tests __call__ method of Pendulum class. Last sub-test tests whether the 
    pendulum is at rest in the equilibrium position."""
    pend = Pendulum(L,M)
    calculated = pend(0,(theta,omega))
    tol = 10**5
    err_msg = lambda s: f"Derivative of {s} off by amount larger than tolerance"
    assert abs(calculated[0]-expected[0]) < tol, err_msg("theta")
    assert abs(calculated[1]-expected[1]) < tol, err_msg("omega")


@pytest.mark.parametrize("prop", [("t"), ("theta"), ("omega")])
def test_pendulum_properties_error(prop):
    """Tests that property access before solve method raises AttributeError."""
    with pytest.raises(AttributeError):
        pend = Pendulum()
        eval(f"pend.{prop}")


def test_initial00_zeroarray():
    """Tests that the correct arrays are returned from solve method for initial 
    values 0,0."""
    pend = Pendulum()
    dt = 0.01
    T = 6
    pend.solve((0,0),T,dt)
    assert np.all(pend.theta == 0), "theta array does not contain only zeroes"
    assert np.all(pend.omega == 0), "omega array does not contain only zeroes"
    assert np.all(pend.t == np.array([i*dt for i,j in enumerate(pend.t)])), \
           "t array wrongly constructed"