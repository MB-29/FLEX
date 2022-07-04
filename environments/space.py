from re import S
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

R = 1
K = 1
period = 2*np.pi*R**(3/2)
omega = 2*np.pi / period
alpha  = 0

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


n_points = 50
interval = torch.linspace(-2*R, 2*R, n_points)
grid_x, grid_y = torch.meshgrid(interval, interval)
grid = torch.cat([
    grid_x.reshape(-1, 1),
    grid_y.reshape(-1, 1),
    # torch.randn(n_points*n_points, 2)
], 1)
x0 = np.array([R, 0, 0,  R*omega])


def test_error(model, x, u, plot, t=0):
    predictions = model.prediction(grid)
    truth = f_star(grid)
    loss_function = nn.MSELoss(reduction='mean')
    if plot and t%10 == 0:
        plot_portrait(predictions)
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
    plt.xlim((-2*R, 2*R))
    plt.ylim((-2*R, 2*R))
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$f_\theta(x,y)$')
