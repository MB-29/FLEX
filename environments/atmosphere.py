from re import S
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

R = 1
K = 1
period = 2*np.pi*(R**3/K)**(1/2)
omega = 2*np.pi / period
alpha  = 0.1*K

def dynamics(x, u):
    # print(np.linalg.norm(u))
    dx = np.zeros_like(x)
    dx[0] = x[1]
    dx[2] = x[3]
    r_2 = x[0]**2 + x[2]**2
    v = np.sqrt(x[1]**2 + x[3]**2)
    dx[1] = -(K / r_2**(3/2)) * x[0] - alpha*x[1] + u[0]
    dx[3] = - (K / r_2**(3/2)) * x[2] - alpha*x[3] + u[1]
    return dx

def f_star_r(r):
    return - K / r**3
# def f_star_v(v):
#     return - alpha *torch.linalg.norm(v, dim=1).unsqueeze(1)*v 

v_max = R* omega

n_points = 20
# interval_x = torch.linspace(-R, R, n_points)
# interval_y = torch.linspace(-R, R, n_points)
interval_r = torch.linspace(0.2*R, 1*R, n_points).unsqueeze(1)
interval_v = torch.linspace(-1*v_max, 1*v_max, n_points)
grid_vx, grid_vy = torch.meshgrid(
    interval_v,
    interval_v,
    # interval_v,
    # interval_v
    )
grid = torch.cat([
    grid_vx.reshape(-1, 1),
    grid_vy.reshape(-1, 1),
    # grid_vx.reshape(-1, 1),
    # grid_vy.reshape(-1, 1),
    # torch.randn(n_points*n_points, 2)
], 1)
x0 = np.array([R, 0, 0,  R*omega])


def test_error(model, x, u, plot, t=0):
    loss_function = nn.MSELoss(reduction='mean')
    predictions_r = model.r_net(interval_r)
    truth_r = f_star_r(interval_r) 
    loss_r = loss_function(predictions_r, truth_r) / (K**2 / R**4)
    # predictions_v = model.v_net(grid)
    # truth_v = f_star_v(grid) 
    # loss_v = loss_function(predictions_v, truth_v) / (alpha**2 * v_max**4)
    # plot_predictions = model.prediction(grid)
    if plot and t%10 == 0:
        # plot_portrait(plot_predictions, t)
        plot_planet(x, t)
        # plt.savefig(f'figures/space-animation/space_{t}.pdf')
        plt.pause(0.1)
        plt.close()
    return loss_r 

def plot_portrait(predictions, t=None):
    vector_x = predictions[:, 0].reshape(n_points, n_points).detach().numpy()
    vector_y = predictions[:, 1].reshape(n_points, n_points).detach().numpy()
    magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
    linewidth = magnitude / magnitude.max()

def plot_planet(x, t=None):
    plt.scatter(x[0], x[2], label='ship', marker="2", s=200)
    plt.scatter(0, 0, marker='*', s=200)
    plt.xlim((-R, R))
    plt.ylim((-R, R))
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(f't = {t}')
    # plt.title(r'$f_\theta(x,y)$')
