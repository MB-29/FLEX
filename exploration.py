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
    """Run exploration

    :type environment: environment object
    :type agent: agent object
    :param T: time horizon
    :type T: int
    :type evaluation: evaluation object
    :param T_random: number of time steps for which random inputs are played in the beginning,
        defaults to 0
    :type T_random: int, optional
    :param reset: reset the environment before running exploration, defaults to True
    :type reset: bool, optional
    :param plot: plot the system, defaults to False
    :type plot: bool, optional
    :param animate: animation function, defaults to None
    :type animate: function, optional
    :param save_models: path to save model parameters, defaults to None
    :type save_models: str, optional
    :return: state-action values, evaluation values
    :rtype: array of shape T x (d+m),  array of size T
    """


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
    from environments.pendulum import DampedPendulum
    from models.pendulum import Linear
    from evaluation.pendulum import ParameterNorm
    from policies import Random, Flex
    from exploration import exploration

    T = 300
    dt = 5e-2


    environment = DampedPendulum(dt)
    model = Linear(environment)
    evaluation = ParameterNorm(environment)

    # agent = Random(
    agent = Flex(
    model,
    environment.d,
    environment.m,
    environment.gamma,
    dt=environment.dt
    )

    z_values, error_values = exploration(environment, agent, T, evaluation, plot=False)