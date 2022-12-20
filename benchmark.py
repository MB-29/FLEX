import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from tqdm import tqdm
import pickle

from agents import Random, Passive
from active_agents import GradientDesign, Spacing, Variation, D_optimal
from environments import get_environment
from exploration import exploration

# ENVIRONMENT_NAME = 'aircraft'
ENVIRONMENT_NAME = 'arm'
ENVIRONMENT_NAME = 'cartpole'
ENVIRONMENT_NAME = 'pendulum'
ENVIRONMENT_NAME = 'gym_cartpole'
ENVIRONMENT_NAME = 'damped_pendulum'
ENVIRONMENT_NAME = 'quadrotor'
ENVIRONMENT_NAME = 'dm_pendulum'
# ENVIRONMENT_NAME = 'damped_pendulum'
# ENVIRONMENT_NAME = 'dm_cartpole'
# ENVIRONMENT_NAME = 'damped_cartpole'

ENVIRONMENT_PATH = f'environments.{ENVIRONMENT_NAME}'
MODEL_PATH = f"models.{ENVIRONMENT_NAME.split('_')[-1]}"
ORACLE_PATH = f'oracles.{ENVIRONMENT_NAME}'

Environment = get_environment(ENVIRONMENT_NAME)
# Environment = importlib.import_module(ENVIRONMENT_PATH).GymPendulum
models = importlib.import_module(MODEL_PATH)
Model = models.LinearA
Model = models.LinearTheta

T = 500
T_random = 0

n_samples = 50

environment = Environment(dt=2e-2)
dt = environment.dt
gamma = environment.gamma
sigma = environment.sigma


from oracles.cartpole import PeriodicOracle

# for Agent in [Random, Active]:
agents = {
    # 'passive':{'agent': Passive, 'color': 'black'},
    # 'periodic': {'agent': PeriodicOracle, 'color': 'blue'},
    'D-optimal': {'agent': D_optimal, 'color': 'blue'},
    'random': {'agent': Random, 'color': 'red'},
    # 'passive': {'agent': Passive, 'color': 'black'},
    # 'uniform': {'agent': Spacing, 'color': 'green'},
    # # 'variation': {'agent': Variation, 'color': 'color'},
    }
output = {}
for name, value in agents.items():
    print(f'agent {name}')
    Agent = value['agent']
    color = value['color']
    # print(f'Model = {Model}')
# for Agent in [Spacing]:
    error_values = np.zeros((n_samples, T))
    for sample_index in tqdm(range(n_samples)): 

        environment.reset()
        model = Model(environment)
        evaluation = model.evaluation
        agent = Agent(
            model,
            environment.d,
            environment.m,
            gamma
            )

        z_values, sample_error = exploration(
            environment, agent, evaluation, T)
        error_values[sample_index, :] = sample_error
    output[name] = error_values

    test_mean = np.mean(error_values, axis=0)
    test_yerr = 2 * np.sqrt(np.var(error_values, axis=0) / n_samples)
    plt.plot(test_mean, alpha=0.7, label=name)
    plt.yscale('log')
    plt.fill_between(np.arange(T), test_mean-test_yerr,
                     test_mean+test_yerr, alpha=0.5)

with open(f'output/{ENVIRONMENT_NAME}_benchmark.pkl', 'wb') as output_file:
    pickle.dump(output, output_file)

plt.legend()
plt.title(r'Test loss')
plt.savefig(f'output/{ENVIRONMENT_NAME}_benchmark.pdf')
plt.show()
