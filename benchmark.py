import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from tqdm import tqdm
import pickle

from agents import Random, Passive
from active_agents import GradientDesign, Spacing, Variation, Linearized

ENVIRONMENT_NAME = 'quadrotor'
# ENVIRONMENT_NAME = 'aircraft'
ENVIRONMENT_NAME = 'arm'
ENVIRONMENT_NAME = 'cartpole'
ENVIRONMENT_NAME = 'pendulum'

ENVIRONMENT_PATH = f'environments.{ENVIRONMENT_NAME}'
MODEL_PATH = f'models.{ENVIRONMENT_NAME}'
ORACLE_PATH = f'oracles.{ENVIRONMENT_NAME}'

environment = importlib.import_module(ENVIRONMENT_PATH)
models = importlib.import_module(MODEL_PATH)
DefaultModel = models.NeuralModel


rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])



T = environment.T
# T = 60
dt = environment.dt
gamma = environment.gamma
sigma = environment.sigma

x0 = environment.x0

n_samples = 100
# for agent_ in [Random, Active]:
agents = {
    # 'passive':{'agent': Passive, 'color': 'black'},
    'random': {'agent': Random, 'color': 'red'},
    'uniform': {'agent': Spacing, 'color': 'green'},
    # # 'gradientOD': {'agent': GradientDesign, 'color': 'purple'},
    # # 'variation': {'agent': Variation, 'color': 'color'},
    'D-optimal': {'agent': Linearized, 'color': 'blue'}
    }
try:
    oracles = importlib.import_module(ORACLE_PATH).oracles
    for name, oracle_ in oracles.items():
        agents[name] = {'agent': oracle_, 'color': 'black'}
        print(f'imported oracle {name}')
except ModuleNotFoundError:
    print('No Oracle found')

# for name in ['explorer', 'passive', 'random', 'periodic']:
# for name in ['linearized']:
# for name in ['od', 'variation']:
output = {}
for name, value in agents.items():
    print(f'agent {name}')
    agent_ = value['agent']
    color = value['color']
    Model = getattr(agent_, 'Model', DefaultModel)
    # print(f'Model = {Model}')
# for agent_ in [Spacing]:
    test_values = np.zeros((n_samples, T))
    for sample_index in tqdm(range(n_samples)):
        model = Model()
        agent = agent_(
            x0.copy(),
            environment.m,
            environment.dynamics,
            model,
            gamma,
            dt
            )

        test_values[sample_index, :] = agent.identify(
            T, test_function=environment.test_error)
    output[name] = test_values

    test_mean = np.mean(test_values, axis=0)
    test_yerr = 2 * np.sqrt(np.var(test_values, axis=0) / n_samples)
    plt.plot(test_mean, alpha=0.7, label=name)
    plt.fill_between(np.arange(T), test_mean-test_yerr,
                     test_mean+test_yerr, alpha=0.5)

with open(f'output/{ENVIRONMENT_NAME}_benchmark.pkl', 'wb') as output_file:
    pickle.dump(output, output_file)

plt.legend()
plt.title(r'Test loss')
plt.savefig(f'output/{ENVIRONMENT_NAME}_benchmark.pdf')
plt.show()
