import torch
import torch.nn as nn
from tqdm import tqdm

from mpc import mpc
from mpc.mpc import QuadCost, GradMethods

import numpy as np
import matplotlib.pyplot as plt

from environments import get_environment


class TransitionModel(nn.Module):
    def __init__(self, model_dynamics, environment, dt):
        self.dynamics = model_dynamics
        self.environment = environment
        self.dt = dt
        super().__init__()

    def forward(self, obs, u):
        x = self.environment.get_state(obs)
        z = torch.cat((x, u), dim=1)
        d_x = self.dynamics(z)
        x_ = x + self.dt*d_x
        obs_ = self.environment.observe(x_)
        return obs_

def get_quadcost(goal_weights, goal_state, R, H, m):
    q = torch.cat((
        goal_weights,
        R*torch.ones(m)
    ))
    px = -torch.sqrt(goal_weights)*goal_state
    p = torch.cat((px, torch.zeros(m)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        H, 1, 1, 1
    )
    p = p.unsqueeze(0).repeat(H, 1, 1)
    quadcost = QuadCost(Q, p)
    return quadcost

def planning(obs, u_init, transition_model, goal_weights, goal_state, R, gamma, lqr_iter):
    H, _, m = u_init.shape
    # print(f"H = {H}, lqr iter = {lqr_iter}, R ={R}")
    # print(f"u_init = {u_init.shape}")
    quadcost = get_quadcost(goal_weights, goal_state, R, H, m)
    nominal_states, nominal_actions, nominal_objs = mpc.MPC(
        5, 1, H,
        u_init=u_init,
        u_lower=-gamma, u_upper=gamma,
        lqr_iter=lqr_iter,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        # linesearch_decay=transition_model.linesearch_decay,
        # max_linesearch_iter=transition_model.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        backprop=False,
        # eps=1e-2,
    )(obs, quadcost, transition_model)
    return nominal_actions


def exploit(environment, model_dynamics, dt, T, mpc_H, lqr_iter, plot=False):
    # print(f"mpc_H = {mpc_H},")
    dynamics = environment.d_dynamics
    gamma = environment.gamma
    m = environment.m
    period = environment.period

    transition_model = TransitionModel(model_dynamics, environment, dt)
    transition = TransitionModel(dynamics, environment, dt)

    goal_weights, goal_state = environment.goal_weights, environment.goal_state
    goal_weights_relaxed = environment.goal_weights_relaxed
    R = environment.R
    
    # u_init = gamma* torch.randn(H, 1, 1)
    # print(u_init)

    obs = environment.observe(torch.tensor(
        environment.x0, dtype=torch.float).unsqueeze(0))
    u_periodic = gamma * \
        torch.sign(torch.cos(2*np.pi*torch.arange(T)
                             * dt/(0.3*period))).view(T, 1, 1)

    first_guess = torch.zeros(T, 1, 1)
    first_guess[:mpc_H, :, :] = planning(obs, u_periodic[:mpc_H], transition_model,
                                         goal_weights_relaxed, goal_state, R, gamma, lqr_iter=100)
    u_init = first_guess.clone()

    cost_values = np.zeros(T)
    for t in range(T):
        next_action = first_guess[t]       
        if lqr_iter is not None:
            goal_weights_t = (t/T) * goal_weights + (1-t/T)*goal_weights_relaxed
            nominal_actions = planning(
                obs, u_init[:mpc_H], transition_model, goal_weights_t, goal_state, R, gamma, lqr_iter)
            next_action = nominal_actions[0]
            u_init = torch.cat((nominal_actions[1:], torch.zeros(1, 1, m)), dim=0)

        # u_init[-2] = u_init[-3]
        # print(f'obs : {obs.shape}')
        # print(f'goal_state : {goal_state.shape}')
        # print(f'next_action : {next_action.shape}')
        z_tilde = torch.cat((obs-goal_state, next_action), dim=1)

        C = torch.diag(torch.tensor(
            [100., 0.1, 100, .1, .1, .1], dtype=torch.float))
        cost = torch.sum(z_tilde*(C@z_tilde.T).T, axis=1)
        cost_values[t] = cost.detach()

        obs = transition(obs, next_action)
        y = obs[0, 0]
        if abs(y) > 1.8:
            obs[0, 0] *= 1.8/abs(y)
            obs[0, 1] = 0

        if not plot:
            continue
            # print(f'obs = {obs.shape}')
        state = environment.get_state(obs[0].unsqueeze(0))
        # print(f'state = {state}')
        environment.plot_system(
            state.squeeze().detach().numpy(),
            next_action[0].detach().numpy(),
            0)
        # axs[i].get_xaxis().set_visible(False)
        # axs[i].get_yaxis().set_visible(False)
        plt.title(f't = {t}')
        plt.pause(0.1)
        plt.close()

    return cost_values


if __name__ == '__main__':
    plot = False
    plot = True

    ENVIRONMENT_NAME = 'gym_cartpole'
    ENVIRONMENT_NAME = 'dm_cartpole'
    ENVIRONMENT_PATH = f'environments.{ENVIRONMENT_NAME}'
    Environment = get_environment(ENVIRONMENT_NAME)

    T = 100
    mpc_H = 100
    lqr_iter = None

    environment = Environment()
    dt = environment.dt
    model_dynamics = environment.d_dynamics
    # model_dynamics = model.forward
    
    
    print('exploit')
    cost_values = exploit(environment, model_dynamics,
                          dt, T, mpc_H, lqr_iter=lqr_iter, plot=True)
    # cost_values = exploit(environment, model_dynamics, u_periodic,
    #                       dt, T, H, lqr_iter, mpc=False, plot=True)

    plt.subplot(211)
    plt.plot(cost_values.cumsum())
    plt.subplot(212)    
    plt.plot(cost_values)
    plt.show()
    print(f'total cost {cost_values.sum()}')
