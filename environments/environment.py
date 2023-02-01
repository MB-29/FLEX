import importlib
import numpy as np

def get_environment(name):
    environment_name = ''.join(word.title() for word in name.split('_'))
    file_name = name.split('_')[-1]
    environment_path = f'environments'
    print(f'file name {file_name}')
    Environment = getattr(importlib.import_module(environment_path), environment_name)
    return Environment

class Environment:

    def __init__(self, x0, d, m, dt, sigma, gamma):
        """
        :param x0: Initial state
        :type x0: array of size d
        :param d: state space dimension
        :type d: int
        :param m: action space dimension
        :type m: int
        :param dt: time step
        :type dt: float
        :param sigma: noise size
        :type sigma: float
        :param gamma: input amplitude
        :type gamma: float
        """
        self.d = d
        self.m = m
        self.dt = dt
        self.sigma = sigma
        self.gamma = gamma
        self.x0 = x0.copy()

        self.x = x0.copy()
    
    def dynamics(self, x, u, t=None):
        """Flow of the dynamics.

        :param x: state
        :type x: array of size d
        :param u: action
        :type u: array of size m
        :param t: time
        :type t: float, optional
        :raises NotImplementedError: _description_
        """
        raise NotImplementedError

    def step(self, u, t):
        """Compute the state increment as a function of the action and time.
        """
        x_dot = self.dynamics(self.x, u, t)
        dx = x_dot * self.dt
        noise =  self.sigma * np.random.randn(self.d)
        dx += noise
        self.x += dx
        self.x = np.clip(self.x, self.x_min, self.x_max)
        return dx

    def reset(self):
        self.x = self.x0.copy()

    
    def d_dynamics(self, x, u):
        raise NotImplementedError
    
    def plot_system(self, x, u):
        raise NotImplementedError