import os
import importlib

def get_environment(name):
    environment_name = ''.join(word.title() for word in name.split('_'))
    environment_path = f'environments'
    Environment = getattr(importlib.import_module(environment_path), environment_name)
    return Environment

class Environment:

    def __init__(self, d, m, dt, sigma, gamma):
        self.d = d
        self.m = m
        self.dt = dt
        self.sigma = sigma
        self.gamma = gamma
    
    def dynamics(self, x, u):
        raise NotImplementedError
    
    def d_dynamics(self, x, u):
        raise NotImplementedError
    
    def test_error(self, model, x, u, plot=False):
        raise NotImplementedError

    def step_cost(self, x, u):
        raise NotImplementedError

    def plot(self, x, u):
        raise NotImplementedError
