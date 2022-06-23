from re import S
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

R = 1
K = 1
period = 2*np.pi*R**(3/2)
omega = 2*np.pi / period
alpha  = 0.1

def dynamics(x, u):
    dx = np.zeros_like(x)
    dx[0] = x[1]
    r_2 = x[0]**2 + x[2]**2
    dx[1] = -(K / r_2**(3/2)) * x[0] - alpha*x[1] + u[0]
    dx[2] = x[3]
    dx[3] = - (K / r_2**(3/2)) * x[2] - alpha*x[3] + u[1]
    return dx

def f_star(x):
    dx = torch.zeros_like(x)
    r_2 = x[:, 0]**2 + x[:, 1]**2
    dx[:, 0] = -(K / r_2**(3/2)) * x[:, 0] 
    dx[:, 1] = -(K / r_2**(3/2)) * x[:, 1]
    return dx

v_max = R**2 * omega**2

n_points = 50
interval_x = torch.linspace(-R, R, n_points)
interval_y = torch.linspace(-R, R, n_points)
interval_r = torch.linspace(R/2, R, n_points)
interval_theta = torch.linspace(0, 2*np.pi, n_points)
grid_x, grid_y = torch.meshgrid(
    interval_x,
    interval_y,
    # interval_v,
    # interval_v
    )
grid_r, grid_theta = torch.meshgrid(
    interval_r,
    interval_theta,
    # interval_v,
    # interval_v
    )
grid = torch.cat([
    grid_x.reshape(-1, 1),
    grid_y.reshape(-1, 1),
    # grid_vx.reshape(-1, 1),
    # grid_vy.reshape(-1, 1),
    # torch.randn(n_points*n_points, 2)
], 1)
disk  = torch.cat([
    grid_r.reshape(-1, 1)*torch.cos(grid_theta.reshape(-1,1)),
    grid_r.reshape(-1, 1)*torch.sin(grid_theta.reshape(-1,1)),
    # grid_y.reshape(-1, 1),
    # grid_vx.reshape(-1, 1),
    # grid_vy.reshape(-1, 1),
    # torch.randn(n_points*n_points, 2)
], 1)
x0 = np.array([R, 0, 0,  R*omega])


def test_error(model, x, u, plot, t=0):
    predictions = model.prediction(disk)
    plot_predictions = model.prediction(grid)
    # predictions /= torch.linalg.norm(predictions, dim=1).unsqueeze(1)
    truth = f_star(disk) 
    # truth /= torch.linalg.norm(truth, dim=1).unsqueeze(1)
    loss_function = nn.MSELoss(reduction='mean')
    if plot and t%10 == 0:
        plot_portrait(plot_predictions)
        plot_planet(x)
        # plt.savefig(f'figures/space-animation/space_{t}.pdf')
        plt.pause(0.1)
        plt.close()
    loss = loss_function(predictions, truth)
    return loss


def plot_portrait(predictions):
    vector_x = predictions[:, 0].reshape(n_points, n_points).detach().numpy()
    vector_y = predictions[:, 1].reshape(n_points, n_points).detach().numpy()
    magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
    linewidth = magnitude / magnitude.max()
    plt.streamplot(
        grid_x.numpy().T,
        grid_y.numpy().T,
        vector_x.T,
        vector_y.T,
        color='black',
        linewidth=linewidth)

def plot_planet(x):
    plt.scatter(x[0], x[2], label='ship', marker="2", s=200)
    plt.scatter(0, 0, marker='*', s=200)
    plt.xlim((-R, R))
    plt.ylim((-R, R))
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$f_\theta(x,y)$')
