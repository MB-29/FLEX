import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from tqdm import tqdm
import pickle

from agents import Random, MaxRandom
from active_agents import D_optimal
from environments.cartpole import DmCartpole as Environment
# from environments.pendulum import GymPendulum as Environment
from models.cartpole import NeuralA as Model
from models.cartpole import NeuralAB as Model
# from models.pendulum import LinearA as Model
from exploration import exploration
# from exploit_cartpole import exploit
from planning_cartpole import exploit

ENVIRONMENT_NAME = 'dm_cartpole'

T = 100
H, lqr_iter = 100, 5
# H, lqr_iter = 100, None
n_samples = 4
n_episodes = 8
sigma = 0.02

environment = Environment(sigma = sigma)
gamma = environment.gamma
dt = environment.dt
x0 = environment.x0


# for Agent in [Random, Active]:
agents = {
    # 'passive':{'agent': Passive, 'color': 'black'},
    'D-optimal': D_optimal,
    'max_random': MaxRandom,
    'random': Random}
name = 'random'
name = 'D-optimal'
# name = 'max_random'
Agent = agents[name]
output = {'n_samples': n_samples, 'gamma': gamma, 'sigma': sigma, 'H': H, 'lqr_iter': lqr_iter}

if __name__ == '__main__':
    task_id = int(sys.argv[1])
    print(f'agent {name}')
    estimation_values = np.zeros((n_samples, n_episodes*T))
    exploitation_values = np.zeros((n_samples, n_episodes))
    output[name] = {'estimation': estimation_values, 'exploitation':exploitation_values}
    for sample_index in tqdm(range(n_samples)):
        environment.reset()
        model = Model(environment)
        evaluation = model.evaluation
        # print(f'Model = {Model}')
    # for Agent in [Spacing]:
        # model = NeuralA(environment)
        # model = FullLinear(environment)
        # print('exploration')
        agent = Agent(
            model,
            environment.d,
            environment.m,
            gamma
        )
        for episode in range(n_episodes):
            print(f'episode {episode}')
            # T_random = T if episode==0 else 0
            T_random = 0
            model_dynamics = model.forward

            cost_values = exploit(
                environment,
                model_dynamics,
                dt,
                T,
                H,
                lqr_iter=lqr_iter,
                plot=False)
            exploitation_values[sample_index, episode] = cost_values.sum()

            z_values, estimation_error = exploration(
                environment, agent, T, evaluation, T_random=T_random, reset=False)

            estimation_values[sample_index, episode*T:(episode+1)*T] = estimation_error

    output[name]['estimation'] = estimation_values
    output[name]['exploitation'] = exploitation_values
    output[name]['z'] = z_values
    exploitation_mean = exploitation_values.mean(axis=0)
    exploitation_std = np.sqrt(exploitation_values.var(axis=0)/n_samples)
    

OUTPUT_PATH = f'output/{ENVIRONMENT_NAME}_{name}_{n_samples}-samples_episodic-{task_id}.pkl'
with open(OUTPUT_PATH, 'wb') as output_file:
    pickle.dump(output, output_file)

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(exploitation_mean, label=name)
# ax1.fill_between(
#     np.arange(n_episodes),
#     exploitation_mean-exploitation_std,
#     exploitation_mean+exploitation_std,
#     alpha=0.5)

estimation_mean = estimation_values.mean(axis=0)
estimation_std = np.sqrt(estimation_values.var(axis=0)/n_samples)
plt.plot(estimation_mean, label=name)
plt.fill_between(
    np.arange(n_episodes*T),
    estimation_mean-estimation_std,
    estimation_mean+estimation_std,
    alpha=0.5)
# ax2.set_yscale('log')
plt.legend()
plt.title(r'Test loss')
plt.show()
# # plt.savefig(f'output/{ENVIRONMENT_NAME}_benchmark.pdf')
