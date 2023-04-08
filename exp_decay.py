import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


class ExponentialDecay:
    """Class for modelling exponential decay."""
    def __init__(self, a):
        if a < 0:
            raise ValueError("Alpha cannot be negative.")
        else: self.a = a

    def __call__(self, t, u):
        """Return the derivative of function u at time t."""
        a = self.a
        return -a * u

    def solve(self, u0, T, dt):
        """
        Returns the solution of the ODE as a tupple of two arrays (time and 
        function values). u0 gives the initial condition, T the upper boundary
        of the solution range, and dt the step size.
        """
        num = int(np.floor(T/dt))
        end = num*dt
        t_array = np.linspace(0, end, num+1)
        sol = solve_ivp(self, [0,T], [u0], t_eval=t_array)
        return sol.t, sol.y[0]


def main():
    decay_model = ExponentialDecay(0.5)
    t,u = decay_model.solve(10, 10, 0.3)

    plt.plot(t, u)
    plt.show()


if __name__ == "__main__":
    main()