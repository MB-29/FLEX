import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from tqdm import tqdm
import pickle

from agents import Random
from active_agents import D_optimal
from exploitation import exploitation
from environments.pendulum import GymPendulum as Environment
from environments.cartpole import DmCartpole as Environment
# from models.pendulum import FullNeural, FullLinear
from models.cartpole import FullNeural, Partial
from environments import get_environment
# from exploit_pendulum import exploit
from exploit_cartpole import exploit
from evaluation.cartpole import ZGrid as Evaluation

# ENVIRONMENT_NAME = 'gym_cartpole'
ENVIRONMENT_NAME = 'gym_pendulum'
ENVIRONMENT_NAME = 'dm_cartpole'



T = 100
T_task = 100
H = 50
n_samples = 1
n_episodes = 10

environment = Environment()
gamma = environment.gamma
sigma = environment.sigma
dt = environment.dt
x0 = environment.x0

evaluation = Evaluation(environment)

# for Agent in [Random, Active]:
agents = {
    # 'passive':{'agent': Passive, 'color': 'black'},
    'D-optimal': D_optimal,
    'random': Random}
    # 'random': {'agent': Random, 'color': 'red'},
    # 'uniform': {'agent': Spacing, 'color': 'green'},
    # # 'gradientOD': {'agent': GradientDesign, 'color': 'purple'},
    # # 'variation': {'agent': Variation, 'color': 'color'},d
output = {'n_samples': n_samples, 'gamma': gamma, 'sigma': sigma}
name = 'D-optimal'
# name = 'random'
Agent = agents[name]

if __name__ == '__main__':
    task_id = int(sys.argv[1])
    print(f'agent {name}')
    estimation_values = np.zeros((n_samples, n_episodes*T))
    exploitation_values = np.zeros((n_samples, n_episodes))
    output[name] = {'estimation': estimation_values, 'exploitation':exploitation_values}
    for sample_index in tqdm(range(n_samples)):
        # print(f'Model = {Model}')
    # for Agent in [Spacing]:
        model = FullNeural(environment)
        # model = Partial(environment)
        # model = FullLinear(environment)
        # print('exploration')
        agent = Agent(
            x0.copy(),
            environment.m,
            environment.dynamics,
            model,
            gamma,
            dt
        )
        for episode in range(n_episodes):
            print(f'episode {episode}')

            agent.x = x0.copy()
            
            model_dynamics = model.forward
            cost_values = exploit(
                environment,
                model_dynamics,
                dt,
                T_task,
                H,
                plot=False)

            estimation_error = agent.identify(
                T,
                test_function=evaluation.test_error,
                T_random=0
            )
            # output[name] = estimation_values
            estimation_values[sample_index, episode*T:(episode+1)*T] = estimation_error

            # print('exploitation')
            exploitation_values[sample_index, episode] = cost_values.sum()
    output[name]['estimation'] = estimation_values
    output[name]['exploitation'] = exploitation_values
    exploitation_mean = exploitation_values.mean(axis=0)
    exploitation_std = np.sqrt(exploitation_values.var(axis=0)/n_samples)
    

OUTPUT_PATH = f'output/{ENVIRONMENT_NAME}_{name}_{n_samples}-samples_H-{H}_episodic-{task_id}.pkl'
with open(OUTPUT_PATH, 'wb') as output_file:
    pickle.dump(output, output_file)

fig, (ax1, ax2) = plt.subplots(2, 1)
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
plt.legend()
plt.show()
plt.legend()
plt.title(r'Test loss')
plt.savefig(f'output/{ENVIRONMENT_NAME}_benchmark.pdf')
