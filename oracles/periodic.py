import numpy as np

from agents import Agent


class Periodic1D(Agent):

    def __init__(self, x0, m, dynamics, model, gamma, dt, period):
        super().__init__(x0, m, dynamics, model, gamma, dt)
        self.period = period

    def policy(self, x, t):
        if t < self.T_random:
            return self.draw_random_control(t)
        u = self.gamma * np.sign(np.sin(2*np.pi*t*self.dt/self.period))
        return np.array([u])

class Periodic2D(Agent):

    def __init__(self, x0, m, dynamics, model, gamma, dt, period):
        super().__init__(x0, m, dynamics, model, gamma, dt)
        self.period = period

    def policy(self, x, t):
        if t < self.T_random:
            return self.draw_random_control(t)
        u = np.zeros(2)
        u[0] = self.gamma * np.sign(np.sin(2*np.pi*t*self.dt/self.period))
        # u[1] = self.gamma * np.cos(2*np.pi*t*self.dt/self.period)
        return u
