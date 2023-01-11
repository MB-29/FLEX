from oracles.cartpole import PeriodicOracle
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from tqdm import tqdm
import pickle

from agents import Random, MaxRandom
from active_agents import D_optimal
from environments.cartpole import DmCartpole as Environment
from models.cartpole import NeuralAB as Model
from models.cartpole import NeuralA as Model
# from models.cartpole import RFF as Model
from exploration import exploration
# from exploit_cartpole import exploit
from planning_cartpole import exploit

plot = True
plot = False


T = 100
H, lqr_iter = 100, 5
# H, lqr_iter = None, 1
T_random = 0
n_episodes = 8

environment = Environment()
dt = environment.dt
gamma = environment.gamma


agents = {
    # 'passive':{'agent': Passive, 'color': 'black'},
    'D-optimal': D_optimal,
    'max_random': MaxRandom}
name = 'max_random'
# name = 'D-optimal'
Agent = agents[name]

environment.reset()
model = Model(environment)
evaluation = model.evaluation
agent = Agent(
    model,
    environment.d,
    environment.m,
    gamma
)

Agent = agents[name]
print(f'agent {name}')
estimation_values = np.zeros(n_episodes*T)
    # print(f'Model = {Model}')
    # for Agent in [Spacing]:
# model = FullLinear(environment)
# print('exploration')
for episode in range(n_episodes):
    print(f'episode {episode}')
    # model_path = f'output/models/pendulum/{name}_episode-{episode}.dat'
    # torch.save(model, model_path)

    z_values, estimation_error = exploration(
        environment, agent, T, evaluation)

    estimation_values[episode * T:(episode+1)*T] = estimation_error

        # print('exploitation')


plt.plot(estimation_values, label=name)
plt.show()

model_dynamics = model.forward
print('exploit')
cost_values = exploit(environment, model_dynamics, dt, T, H, lqr_iter=lqr_iter, plot=True)
print(f'final cost {cost_values.sum()}')
plt.plot(cost_values.cumsum())
plt.legend()
plt.show()
# OUTPUT_PATH = f'output/{ENVIRONMENT_NAME}_{name}_{n_samples}-samples_episodic-{task_id}.pkl'
# with open(OUTPUT_PATH, 'wb') as output_file:
#     pickle.dump(output, output_file)
# plt.legend()
# plt.title(r'Test loss')
# plt.savefig(f'output/{ENVIRONMENT_NAME}_benchmark.pdf')
