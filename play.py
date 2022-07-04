import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from agents import Random, Passive
from active_agents import OptimalDesign, Spacing, Variation, Linearized
from oracles.cartpole import Oracle

ENVIRONMENT_NAME = 'quadrotor'
# ENVIRONMENT_NAME = 'quadrotor'
# ENVIRONMENT_NAME = 'pendulum'

ENVIRONMENT_PATH = f'environments.{ENVIRONMENT_NAME}'
MODEL_PATH = f'models.{ENVIRONMENT_NAME}'
ORACLE_PATH = f'oracles.{ENVIRONMENT_NAME}'

environment = importlib.import_module(ENVIRONMENT_PATH)
Model = getattr(importlib.import_module(MODEL_PATH), 'Model')
try:
    Oracle = getattr(importlib.import_module(ORACLE_PATH), 'Oracle')
    print('Oracle imported')
except ModuleNotFoundError:
    print('No Oracle found')

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

plot = False
plot = True

T = environment.T
# T = 100
dt = environment.dt
gamma = environment.gamma
sigma = environment.sigma

x0 = environment.x0

agent_ = Random
agent_ = Linearized
# agent_ = Oracle

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
