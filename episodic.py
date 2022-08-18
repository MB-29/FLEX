import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from tqdm import tqdm
import pickle

from agents import Random
from active_agents import Linearized
from exploitation import exploitation

# ENVIRONMENT_NAME = 'aircraft'
ENVIRONMENT_NAME = 'pendulum_gym'

ENVIRONMENT_PATH = f'environments.{ENVIRONMENT_NAME}'
MODEL_PATH = f'models.{ENVIRONMENT_NAME}'
ORACLE_PATH = f'oracles.{ENVIRONMENT_NAME}'

environment = importlib.import_module(ENVIRONMENT_PATH)
models = importlib.import_module(MODEL_PATH)
DefaultModel = models.NeuralModel

T = environment.T
dt = environment.dt
gamma = environment.gamma
sigma = environment.sigma

x0 = environment.x0

n_samples = 2
T_random = 0
n_episodes = 5
# for agent_ in [Random, Active]:
agents = {
    # 'passive':{'agent': Passive, 'color': 'black'},
    'D-optimal': {'agent': Linearized, 'color': 'blue'},
    'random': {'agent': Random, 'color': 'red'},
    # 'uniform': {'agent': Spacing, 'color': 'green'},
    # # 'gradientOD': {'agent': GradientDesign, 'color': 'purple'},
    # # 'variation': {'agent': Variation, 'color': 'color'},
}
output = {}

fig, (ax1, ax2) = plt.subplots(2, 1)

for name, value in agents.items():
    print(f'agent {name}')


    estimation_values = np.zeros((n_samples, n_episodes*T))
    exploitation_values = np.zeros((n_samples, n_episodes))
    output[name] = {'estimation': estimation_values, 'exploitation':exploitation_values}
    for sample_index in tqdm(range(n_samples)):
        agent_ = value['agent']
        color = value['color']
        Model = getattr(agent_, 'Model', DefaultModel)
        # print(f'Model = {Model}')
    # for agent_ in [Spacing]:
        model = Model()
        for episode in range(n_episodes):
            agent = agent_(
                x0.copy(),
                environment.m,
                environment.dynamics,
                model,
                gamma,
                dt
            )
            # print('exploration')
            estimation_error = agent.identify(
                T,
                test_function=environment.test_error,
                T_random=T_random
            )
            # output[name] = estimation_values
            estimation_values[sample_index, episode*T:(episode+1)*T] = estimation_error

            # print('exploitation')
            cost, loss_values = exploitation(model.model, environment.dynamics, n_gradient=500)
            exploitation_values[sample_index, episode] = cost
    output[name]['estimation'] = estimation_values
    output[name]['exploitation'] = exploitation_values
    exploitation_mean = exploitation_values.mean(axis=0)
    exploitation_std = np.sqrt(exploitation_values.var(axis=0)/n_samples)
    ax1.plot(exploitation_mean, label=name)
    ax1.fill_between(
        np.arange(n_episodes),
        exploitation_mean-exploitation_std,
        exploitation_mean+exploitation_std,
        alpha=0.5)
    estimation_mean = estimation_values.mean(axis=0)
    estimation_std = np.sqrt(estimation_values.var(axis=0)/n_samples)
    ax2.plot(estimation_mean, label=name)
    ax2.fill_between(
        np.arange(n_episodes*T),
        estimation_mean-estimation_std,
        estimation_mean+estimation_std,
        alpha=0.5)

with open(f'output/{ENVIRONMENT_NAME}_exploit.pkl', 'wb') as output_file:
    pickle.dump(output, output_file)
plt.legend()
# plt.legend()
# plt.title(r'Test loss')
# plt.savefig(f'output/{ENVIRONMENT_NAME}_benchmark.pdf')
plt.show()
