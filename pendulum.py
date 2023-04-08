"""Module for modeling the motion of single pendulums."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Pendulum:
    """Class for modelling the motion of a single pendulum."""
    def __init__(self, L=1, M=1, g=9.81):
        self.L = L
        self.M = M
        self.g = g

    def __call__(self,t,y):
        """Returns the derivative of theta and omega at time t."""
        theta, omega = y
        return omega, -(self.g/self.L) * np.sin(theta)
        
    def solve(self, y0, T, dt, angle = "rad"):
        """
        Calculates the solution of the ODE. u0 gives the initial condition, 
        T the upper boundary of the solution range, and dt the step size.
        """
        if angle == "deg":
            y0 = np.radians(y0)
        elif angle != "rad":
            raise Exception
        num = int(np.floor(T/dt))
        end = num*dt
        t_array = np.linspace(0, end, num+1)
        sol = solve_ivp(self, [0,T], y0, t_eval=t_array)
        self._t, self._theta, self._omega = sol.t, sol.y[0], sol.y[1]

    @property
    def t(self):
        return self._t

    @property
    def theta(self):
        return self._theta

    @property
    def omega(self):
        return self._omega

    @property
    def x(self):
        return self.L * np.sin(self._theta)   

    @property 
    def y(self):
        return -self.L * np.cos(self._theta)

    @property
    def potential(self):
        return self.M*self.g * (self.y+self.L)

    @property
    def v_x(self):
        dt = self._t[1] - self._t[0]
        return np.gradient(self.x, dt)

    @property
    def v_y(self):
        dt = self._t[1] - self._t[0]
        return np.gradient(self.y, dt)

    @property
    def kinetic(self):
        return (1/2)*self.M * (self.v_x**2+self.v_y**2)


class DampenedPendulum(Pendulum):
    """Class for modeling the motion of a single dampened pendulum."""
    def __init__(self,L=1, M=1, B=1, g=9.81):
        super().__init__(L, M, g)
        self.B = B

    def __call__(self, t, y):
        """Returns the derivative of theta and omega at time t."""
        theta, omega = y
        return omega, -(self.g/self.L)*np.sin(theta) - (self.B/self.M)*omega


def main(pendtype):
    pend = eval(f"{pendtype}()")
    pend.solve((90,0), 6, 0.01, angle="deg")
    plt.figure(figsize=(10,6))
    
    plt.suptitle(pendtype, fontsize = 14)
    plt.subplot(1,2,1)
    plt.title("Plot of angle with time (in radians)")
    plt.ylabel("Angle")
    plt.xlabel("Time")
    plt.plot(pend.t, pend.theta)

    plt.subplot(1,2,2)
    plt.title("Plot of energy with time (in joules)")
    plt.ylabel("Energy")
    plt.xlabel("Time")
    plt.plot(pend.t, pend.kinetic, label="Kinetic energy")
    plt.plot(pend.t, pend.potential, label="Potential energy")
    plt.plot(pend.t, pend.kinetic+pend.potential, label="Total energy")
    plt.legend(loc = "upper right")


if __name__ == "__main__":
    main("Pendulum")
    main("DampenedPendulum")
    plt.show()