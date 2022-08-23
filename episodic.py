import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from tqdm import tqdm
import pickle

from agents import Random
from active_agents import Linearized
from exploitation import exploitation
from environments.gym_pendulum import GymPendulum as Environment
from models.gym_pendulum import NeuralModel, LinearModel
from environments import get_environment


ENVIRONMENT_NAME = 'gym_pendulum'

Environment = get_environment(ENVIRONMENT_NAME)

dt = 80e-4
task_dt = 80e-3

T = 100
T_task = 100
n_samples = 10
T_random = 0
n_episodes = 20
n_gradient = 400

environment = Environment(dt=dt)
exploitation_environment = Environment(dt=task_dt)
gamma = environment.gamma
sigma = environment.sigma

x0 = environment.x0

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
if __name__ == '__main__':
    task_id = int(sys.argv[1])
    for name, value in agents.items():
        print(f'agent {name}')
        estimation_values = np.zeros((n_samples, n_episodes*T))
        exploitation_values = np.zeros((n_samples, n_episodes))
        output[name] = {'estimation': estimation_values, 'exploitation':exploitation_values}
        for sample_index in tqdm(range(n_samples)):
            agent_ = value['agent']
            color = value['color']
            NeuralModel
            # print(f'Model = {Model}')
        # for agent_ in [Spacing]:
            model = NeuralModel()
            # print('exploration')
            for episode in range(n_episodes):
                # print(f'episode {episode}')
                agent = agent_(
                    x0.copy(),
                    environment.m,
                    environment.dynamics,
                    model,
                    gamma,
                    dt
                )
                estimation_error = agent.identify(
                    T,
                    test_function=environment.test_error,
                    T_random=0
                )
                # output[name] = estimation_values
                estimation_values[sample_index, episode*T:(episode+1)*T] = estimation_error

                # print('exploitation')
                cost, loss_values = exploitation(
                    exploitation_environment,
                    model.model,
                    T_task,
                    n_gradient=n_gradient
                    )
                exploitation_values[sample_index, episode] = cost
        output[name]['estimation'] = estimation_values
        output[name]['exploitation'] = exploitation_values
        exploitation_mean = exploitation_values.mean(axis=0)
        exploitation_std = np.sqrt(exploitation_values.var(axis=0)/n_samples)
        # ax1.plot(exploitation_mean, label=name)
        # ax1.fill_between(
        #     np.arange(n_episodes),
        #     exploitation_mean-exploitation_std,
        #     exploitation_mean+exploitation_std,
        #     alpha=0.5)
        # estimation_mean = estimation_values.mean(axis=0)
        # estimation_std = np.sqrt(estimation_values.var(axis=0)/n_samples)
        # ax2.plot(estimation_mean, label=name)
        # ax2.fill_between(
        #     np.arange(n_episodes*T),
        #     estimation_mean-estimation_std,
        #     estimation_mean+estimation_std,
        #     alpha=0.5)
OUTPUT_PATH = f'output/{ENVIRONMENT_NAME}_exploit-{task_id}.pkl'
with open(OUTPUT_PATH, 'wb') as output_file:
    pickle.dump(output, output_file)
plt.legend()
# plt.legend()
# plt.title(r'Test loss')
# plt.savefig(f'output/{ENVIRONMENT_NAME}_benchmark.pdf')
plt.show()
