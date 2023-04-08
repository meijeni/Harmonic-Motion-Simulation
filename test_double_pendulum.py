import pytest
import numpy as np
from double_pendulum import DoublePendulum

@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (  0,   0,            0),
        (  0, 0.5,  3.386187037), 
        (0.5,   0, -7.678514423),
        (0.5, 0.5, -4.703164534),
    ]
)
def test_domega1_dt(theta1, theta2, expected):
    dp = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    dtheta1_dt, domega1_dt, _, _ = dp(t, y)
    assert np.isclose(dtheta1_dt, 0.25)
    assert np.isclose(domega1_dt, expected)
    
@pytest.mark.parametrize(
    "theta1, theta2, expected",
    [
        (  0,   0,          0.0),
        (  0, 0.5, -7.704787325),
        (0.5,   0,  6.768494455),
        (0.5, 0.5,          0.0),
    ],
)
def test_domega2_dt(theta1, theta2, expected):
    dp = DoublePendulum()
    t = 0
    y = (theta1, 0.25, theta2, 0.15)
    _, _, dtheta2_dt, domega2_dt = dp(t, y)
    assert np.isclose(dtheta2_dt, 0.15)
    assert np.isclose(domega2_dt, expected)

def test_initial0000_zeroarray():
    """Tests that the correct arrays are returned from solve method for initial 
    values 0,0,0,0."""
    dp = DoublePendulum()
    dt = 0.01
    T = 6
    dp.solve((0,0,0,0), T, dt)
    assert np.all(dp.theta1 == 0), "theta1 array does not contain only zeroes"
    assert np.all(dp.omega1 == 0), "omega1 array does not contain only zeroes"
    assert np.all(dp.theta2 == 0), "theta2 array does not contain only zeroes"
    assert np.all(dp.omega2 == 0), "omega2 array does not contain only zeroes"
    assert np.all(dp.t == np.array([i*dt for i,j in enumerate(dp.t)])), \
           "t array wrongly constructed"

@pytest.mark.parametrize("theta1, omega1, theta2, omega2",[
                        (90,0,0,0), (0,10,0,0), (100,20,5,5), (0,0,90,0)])
def test_energy_conserved(theta1, omega1, theta2, omega2,):
    """Tests that the solve method for DoublePendulum conserves energy."""
    dp = DoublePendulum()
    dt = 0.01
    T = 6
    dp.solve((0,0,0,0), T, dt, angle="deg")
    assert np.all(np.isclose(dp.potential+dp.kinetic,
                             dp.potential[0]+dp.kinetic[0])
                 )

@pytest.mark.parametrize("prop",[
                        ("t"), ("theta1"), ("omega1"), ("theta2"), ("omega2"),
                        ("x1"), ("y1"), ("x2"), ("y2")])
def test_pendulum_properties_error(prop):
    """Tests that property access before solve method raises AttributeError."""
    with pytest.raises(AttributeError):
        pend = DoublePendulum()
        eval(f"pend.{prop}")

