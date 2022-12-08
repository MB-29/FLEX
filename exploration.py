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
        # print(f't = {t}')
        x = environment.x.copy()

        u = agent.policy(x, t) 
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
    # from environments.cartpole import GymCartpole as Environment

    from models.pendulum import LinearA as Model
    # from models.pendulum import LinearTheta as Model
    # from models.cartpole import FullNeural as Model
    from agents import Passive as Agent
    from agents import Random as Agent
    # from active_agents import D_optimal as Agent

    plot = False
    plot = True

    environment = Environment(dt=2e-2)
    model = Model(environment)
    evaluation = model.evaluation

    agent = Agent(
    model,
    environment.d,
    environment.m,
    environment.gamma
    )

    T = 200

    z_values, error_values = exploration(environment, agent, evaluation, T, plot=plot)

    plt.plot(error_values) ; plt.show()