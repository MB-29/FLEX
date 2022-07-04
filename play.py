import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from agents import Random, Passive, OptimalDesign, Spacing, Periodic, Variation, Linearized

ENVIRONMENT_NAME = 'cartpole'
ENVIRONMENT_NAME = 'quadrotor'
# ENVIRONMENT_NAME = 'pendulum'

ENVIRONMENT_PATH = f'environments.{ENVIRONMENT_NAME}'
MODEL_PATH = f'models.{ENVIRONMENT_NAME}'

environment = importlib.import_module(ENVIRONMENT_PATH)
Model = getattr(importlib.import_module(MODEL_PATH), 'Model')

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

plot = False
# plot = True

T = environment.T
# T = 100
dt = environment.dt
gamma = environment.gamma
sigma = environment.sigma

x0 = environment.x0

agent_ = Random
agent_ = Linearized

test_values = np.zeros(T)
model = Model()
agent = agent_(
    x0.copy(),
    environment.m,
    environment.dynamics,
    model,
    gamma,
    dt
    )

test_values = agent.identify(T, test_function=environment.test_error, plot=plot)

plt.plot(test_values, alpha=0.7)
plt.legend()
plt.title(r'Test loss')
plt.show()
