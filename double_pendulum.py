"""Module for modelling the motion of double pendulums."""
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


class DoublePendulum:
    def __init__(self, L1=1, L2=1, g=9.81):
        """Class for modelling the motion of double pendulums."""
        self.L1 = L1
        self.L2 = L2
        self.g = g

    def __call__(self, t, y):
        """
        Returns the derivative of theta1, theta2 and omega1 and omega2 at
        time t.
        """
        theta1, omega1, theta2, omega2 = y
        L1, L2, g = self.L1, self.L2, self.g
        dt = theta2 - theta1
        sin, cos = np.sin, np.cos

        dtheta1_dt = omega1
        domega1_dt = (
                     (
                     L1 * omega1**2 * sin(dt) * cos(dt) 
                     + g * sin(theta2) * cos(dt)
                     + L2 * omega2**2 * sin(dt)
                     - 2*g * sin(theta1)
                     ) 
                     /(2*L1 - L1 * (cos(dt))**2)
                     )
        dtheta2_dt = omega2
        domega2_dt = (
                     (
                     -L2 * omega2**2 * sin(dt) * cos(dt) 
                     + 2*g * sin(theta1) * cos(dt)
                     - 2*L1 * omega1**2 * sin(dt)
                     - 2*g * sin(theta2)
                     ) 
                     /(2*L2 - L2 * (cos(dt))**2)
                     )
                     
        return dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt

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
        sol = solve_ivp(self, [0,T], y0, t_eval=t_array, method="Radau")
        self.dt = dt
        self._t = sol.t
        self._theta1, self._omega1, self._theta2, self._omega2 = sol.y
    
    @property
    def t(self):
        return self._t

    @property
    def theta1(self):
        return self._theta1
    
    @property
    def theta2(self):
        return self._theta2

    @property
    def omega1(self):
        return self._omega1

    @property
    def omega2(self):
        return self._omega2

    @property
    def x1(self):
        return self.L1 * np.sin(self._theta1)   

    @property 
    def y1(self):
        return -self.L1 * np.cos(self._theta1)

    @property
    def x2(self):
        return self.x1 + self.L2*np.sin(self._theta2)   

    @property 
    def y2(self):
        return self.y1 - self.L2*np.cos(self._theta2)

    @property 
    def potential(self):
        P1 = self.g * (self.y1 + self.L1)
        P2 = self.g * (self.y2 + self.L2 + self.L2)
        return P1 + P2

    @property
    def kinetic(self):
        dt = self._t[1] - self._t[0]
        v_x1 = np.gradient(self.x1, dt)
        v_y1 = np.gradient(self.y1, dt)
        v_x2 = np.gradient(self.x2, dt)
        v_y2 = np.gradient(self.y2, dt)
        K1 = (1/2) * (v_x1**2+v_y1**2)
        K2 = (1/2) * (v_x2**2+v_y2**2)
        return K1 + K2

    def create_animation(self):
        """Creates an animation object from the last call of solve method."""
        fig = plt.figure()
            
        plt.axis('equal')
        plt.axis('off')
        plt.axis((-3, 3, -3, 3))
            
        self.pendulums, = plt.plot([], [], 'o-', lw=2)
            
        self.animation = animation.FuncAnimation(fig,
                                                self._next_frame,
                                                frames=range(len(self.x1)), 
                                                repeat=None,
                                                interval=1000*self.dt, 
                                                blit=True)

    def _next_frame(self, i):
        self.pendulums.set_data((0, self.x1[i], self.x2[i]),
                                (0, self.y1[i], self.y2[i]))
        return self.pendulums,

    def show_animation(self):
        """Shows animation made from last call of create_animation method."""
        try:
            self.animation
        except AttributeError:
            raise AttributeError
        plt.show()

    def save_animation(self, filename):
        """Saves animation made from last call of create_animation method to mp4
        file."""
        try:
            self.animation
        except AttributeError:
            raise AttributeError  
        # using self.animation.save(filename, fps=60) throws error (why????):
        # MovieWriter ffmpeg unavailable; using Pillow instead.
        # --- lots of error stuff ---
        # ValueError: unknown file extension: .mp4
        # Solution: Download ImageMagick and use ImageMagickWriter
        FFwriter = animation.ImageMagickWriter(fps=60)
        self.animation.save(filename, FFwriter)

def main(pendtype):
    pend = eval(f"{pendtype}()")
    pend.solve((90,0,90,0), 6, 0.01, angle="deg")
    plt.figure(figsize=(10,8))
    plt.suptitle(pendtype, fontsize = 14)
    plt.title("Plot of energy with time (in joules)")
    plt.ylabel("Energy")
    plt.xlabel("Time")
    plt.plot(pend.t, pend.kinetic, label="Kinetic energy")
    plt.plot(pend.t, pend.potential, label="Potential energy")
    plt.plot(pend.t, pend.kinetic+pend.potential, label="Total energy")
    plt.legend(loc = "upper right")

    pend.solve((90,0,90,0), 10, 1/60, angle="deg")
    pend.create_animation()
    pend.show_animation()
    pend.save_animation("example_simulation.mp4")


if __name__ == "__main__":
    main("DoublePendulum")

