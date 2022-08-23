import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from agents import Random, Passive
from active_agents import GradientDesign, Spacing, Variation, Linearized

from environments import get_environment

ENVIRONMENT_NAME = 'aircraft'
ENVIRONMENT_NAME = 'quadrotor'
ENVIRONMENT_NAME = 'arm'
ENVIRONMENT_NAME = 'pendulum'
ENVIRONMENT_NAME = 'gym_pendulum'
# ENVIRONMENT_NAME = 'cartpole'

ENVIRONMENT_PATH = f'environments.{ENVIRONMENT_NAME}'
MODEL_PATH = f'models.{ENVIRONMENT_NAME}'
ORACLE_PATH = f'oracles.{ENVIRONMENT_NAME}'
Environment = get_environment(ENVIRONMENT_NAME)
# Environment = importlib.import_module(ENVIRONMENT_PATH).GymPendulum
models = importlib.import_module(MODEL_PATH)

plot = False
plot = True

T = 100
environment = Environment(80e-4)
dt = environment.dt
gamma = environment.gamma
sigma = environment.sigma

x0 = environment.x0

Agent = Random
Agent = Linearized
# Agent = Spacing
# Agent = oracles.LinearOracle

# model = models.Model()
model = models.NeuralModel()
# model = models.LinearModel()

agent = Agent(
    x0.copy(),
    environment.m,
    environment.dynamics,
    model,
    gamma,
    dt
    )

test_values = agent.identify(
    T,
    test_function=environment.test_error,
    plot=plot,
    T_random=0
    )

plt.plot(test_values, alpha=0.7)
plt.legend()
plt.title(r'Test loss')
plt.show()
