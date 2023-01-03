import numpy as np
import matplotlib.pyplot as plt

def exploration(
    environment,
    agent,
    evaluation, 
    T,
    T_random=0,
    plot=False):

    d, m = environment.d, environment.m
    z_values = np.zeros((T, d+m))
    error_values = np.zeros(T)  

    for t in range(T):
        # if t%100 == 0:
        #     environment.reset()
        # print(f't = {t}')
        x = environment.x.copy()

        u = agent.policy(x, t) if t>T_random else agent.draw_random_control()
        # u = 0*u if t>20 else u
        dx = environment.step(u)
        dx_dt = dx/environment.dt
        agent.learning_step(x, dx_dt, u)

        z_values[t:, :d] = x.copy()
        z_values[t:, d:] = u.copy()

        error_values[t] = evaluation.evaluate(agent.model)

        if plot:
            environment.plot_system(x, u, t)
            plt.pause(0.1)
            plt.close()

    return z_values, error_values

if __name__=='__main__':
    from environments.pendulum import DampedPendulum as Environment
    from environments.pendulum import DmPendulum as Environment
    # from environments.pendulum import GymPendulum as Environment
    from models.pendulum import LinearA as Model
    from models.pendulum import LinearTheta as Model

    from environments.cartpole import GymCartpole as Environment
    # from models.cartpole import RFF as Model
    from models.cartpole import NeuralA as Model
    from models.cartpole import NeuralAB as Model
    # from models.cartpole import Neural as Model

    from agents import Passive as Agent
    from agents import Random as Agent
    from agents import MaxRandom as Agent
    from active_agents import D_optimal as Agent
    # from oracles.cartpole import PeriodicOracle as Agent

    plot = False
    # plot = Tru<e

    environment = Environment()
    # environment = Environment(0.08)
    model = Model(environment)
    evaluation = model.evaluation

    agent = Agent(
    model,
    environment.d,
    environment.m,
    environment.gamma
    )

    T = 800

    z_values, error_values = exploration(environment, agent, evaluation, T, plot=plot)
    # plt.subplot(211)
    plt.plot(error_values)
    # plt.subplot(212)
    # plt.plot(z_values[:, 2])
    # plt.yscale('log')
    plt.show()