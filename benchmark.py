import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from tqdm import tqdm

from agents import Random, Passive
from active_agents import OptimalDesign, Spacing, Periodic, Variation, Linearized

ENVIRONMENT_NAME = 'cartpole'
ENVIRONMENT_NAME = 'quadrotor'

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



T = environment.T
T = 60
dt = environment.dt
gamma = environment.gamma
sigma = environment.sigma

x0 = environment.x0

n_samples = 5
# for agent_ in [Random, Active]:
agents = {
    'passive':{'agent': Passive, 'color': 'black'},
    'random': {'agent': Random, 'color': 'red'},
    'oracle': {'agent': Periodic, 'color': 'black'},
    'explorer': {'agent': Spacing, 'color': 'orange'},
    'D-optimal': {'agent': OptimalDesign, 'color': 'purple'},
    # 'variation': {'agent': Variation, 'color': 'color'},
    'linearized': {'agent': Linearized, 'color': 'green'}
    }

# for name in ['explorer', 'passive', 'random', 'periodic']:
for name in ['linearized', 'explorer', 'D-optimal', 'random']:
    print(f'agent {name}')
# for name in ['od', 'variation']:
    agent_ = agents[name]['agent']
    color = agents[name]['color']
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

    test_mean = np.mean(test_values, axis=0)
    test_yerr = 2 * np.sqrt(np.var(test_values, axis=0) / n_samples)
    plt.plot(test_mean, alpha=0.7, label=name, color=color)
    plt.fill_between(np.arange(T), test_mean-test_yerr,
                     test_mean+test_yerr, alpha=0.5, color=color)
plt.legend()
plt.title(r'Test loss')
plt.savefig('benchmark.pdf')
plt.show()
