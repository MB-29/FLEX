import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

def exploration(
    environment,
    agent,
    T,
    evaluation, 
    T_random=0,
    reset=True,
    plot=False,
    animate=None,
    save_models=None):


    d, m = environment.d, environment.m
    z_values = np.zeros((T, d+m))
    error_values = np.zeros(T)  

    if reset:
        environment.reset()

    for t in range(T):

        x = environment.x.copy()
        u = agent.draw_random_control()
        if t > T_random:
            u = agent.policy(x, t)

        dx = environment.step(u, t)
        dx_dt = dx/environment.dt
        agent.learning_step(x, dx_dt, u)

        z_values[t:, :d] = x.copy()
        z_values[t:, d:] = u.copy()

        if evaluation is not None:
            error_values[t] = evaluation.evaluate(agent.model, t)
        
        illustrate(x, u, t, agent.model, z_values, error_values, environment.plot_system, plot, animate, save_models)
    return z_values, error_values


def illustrate(x, u, t, model, z_values, error_values, plot_system, plot, animate, save_models):        

        if save_models is not None:
            path = f'{save_models}_{t}.dat'
            with open(path, 'wb') as file:
                torch.save(model, file)
        if animate is not None:
            animate(model, u, t, z_values, error_values, plot=plot)
            return
        if plot:
            plot_system(x, u, t)
            plt.pause(0.1)
            plt.close()


if __name__=='__main__':
    from environments.pendulum import DmPendulum as Environment
    from environments.pendulum import DampedPendulum as Environment
    # from environments.pendulum import GymPendulum as Environment
    from models.pendulum import LinearA as Model
    from models.pendulum import LinearAB as Model

    # from environments.cartpole import GymCartpole as Environment
    # # from models.cartpole import RFF as Model
    # from models.cartpole import NeuralA as Model
    # from models.cartpole import NeuralAB as Model

    from environments.cartpole import DampedCartpole as Environment
    from models.cartpole import NeuralA as Model

    from environments.arm import DampedArm as Environment
    from models.arm import NeuralA as Model

    from environments.quadrotor import DefaultQuadrotor as Environment
    from models.quadrotor import NeuralModel as Model

    # from policies import Passive as Agent
    # from policies import Random as Agent
    from policies import MaxRandom as Agent
    # from oracles.arm import PeriodicOracle as Agent
    from policies import Flex as Agent

    plot = False
    plot = True

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

    T = 400
    z_values, error_values = exploration(environment, agent, T, evaluation, plot=plot)
    # plt.subplot(211)
    plt.plot(error_values)
    # plt.yscale('log')
    plt.show()
    # plt.subplot(212)
    # plt.plot(z_values[:, 1], z_values[:, 3])
    # plt.yscale('log')
    plt.show()