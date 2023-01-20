import numpy as np
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
    animate=None):


    d, m = environment.d, environment.m
    z_values = np.zeros((T, d+m))
    error_values = np.zeros(T)  

    if reset:
        environment.reset()
    for t in range(T):
        # if t%100 == 0:
        #     environment.reset()
        # print(f't = {t}')
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
        if animate is not None:
            animate(agent.model, z_values, u, t, plot=plot)
            continue
        if plot:
            environment.plot_system(x, u, t)
            plt.pause(0.1)
            plt.close()

    return z_values, error_values

if __name__=='__main__':
    from environments.pendulum import DmPendulum as Environment
    from environments.pendulum import DampedPendulum as Environment
    # from environments.pendulum import GymPendulum as Environment
    from models.pendulum import LinearA as Model
    from models.pendulum import LinearTheta as Model

    # from environments.cartpole import GymCartpole as Environment
    # # from models.cartpole import RFF as Model
    # from models.cartpole import NeuralA as Model
    # from models.cartpole import NeuralAB as Model

    from environments.cartpole import DampedCartpole as Environment
    from models.cartpole import NeuralA as Model

    from environments.arm import DampedArm as Environment
    from models.arm import NeuralA as Model

    # # from environments.quadrotor import DefaultQuadrotor as Environment
    # # from models.quadrotor import NeuralModel as Model

    # from environments.potential import DefaultPotential as Environment
    # from models.potential import NeuralModel as Model

    # from agents import Passive as Agent
    # from agents import Random as Agent
    from agents import MaxRandom as Agent
    # from oracles.arm import PeriodicOracle as Agent
    from active_agents import D_optimal as Agent

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
    environment.gamma,
    dt=environment.dt,
    batch_size=100
    )

    T = 500

    z_values, error_values = exploration(environment, agent, T, evaluation, plot=plot, T_random=0)
    # plt.subplot(211)
    plt.plot(error_values)
    # plt.yscale('log')
    plt.show()
    # plt.subplot(212)
    # plt.plot(z_values[:, 1], z_values[:, 3])
    # # plt.yscale('log')
    # plt.show()