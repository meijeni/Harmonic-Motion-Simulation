
from scipy.integrate import solve_ivp
import numpy as np # Task 3
import pytest

# New file exp_decay.py

# Exercise 1a)
class ExponentialDecay:
    def __init__(self, a):
        if a < 0:
            raise ValueError("Alpha cannot be negative.")
        else: self.a = a

    def __call__(self, t, u):
        a = self.a
        return -a * u

    # Exercise 1c) ??????


    def solve(self, u0, T, dt):
        a = self.a
        def model(t, u): return -a * u
        sol = solve_ivp(model, [0, T], [u0])
        t = np.linspace(0, T, dt)
        u = sol.sol(t)
        return t, u
"""
# New file test_exp_decay.py

# Exercise 1b)
#import pytest
# from exp_decay import ExponentialDecay∏


@pytest.mark.parametrize("u, t, a, expected", [(3.2, 1, 0.4, -1.28)]) #Bare legge til en verdi for t?
def test_dudt(u, t, a, expected):
    verify = ExponentialDecay(a)
    calculated = verify(t, u)
    tol = 10**-5
    assert abs(calculated - expected) < tol

def test_negative_a():
    with pytest.raises(ValueError):
            ExponentialDecay(-1)

#Exercise 1c)

"""
import matplotlib as plt
decay_model = ExponentialDecay(0.4)
t, u = decay_model.solve(1, 1, 1)
print(t)
print(u)

plt.plot(t, u)
plt.show()
"""
