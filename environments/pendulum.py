import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

alpha = 1
omega_2 = 1

def dynamics(x, u):
    dx = np.zeros_like(x)
    dx[0] = x[1]
    dx[1] = -omega_2 * np.sin(x[0]) - alpha*x[1] + u
    return dx


def f_star(x):
    dx = torch.zeros_like(x)
    dx[:, 0] = x[:, 1]
    dx[:, 1] = -omega_2 * np.sin(x[:, 0]) - alpha*x[:, 1]
    return dx


n_points = 100
phi_max = np.pi
dphi_max = np.sqrt(-2*omega_2*np.cos(phi_max))
interval_q = torch.linspace(-phi_max, phi_max, n_points)
interval_p = torch.linspace(-dphi_max, dphi_max, n_points)
grid_q, grid_p = torch.meshgrid(interval_q, interval_p)
grid = torch.cat([
    grid_q.reshape(-1, 1),
    grid_p.reshape(-1, 1),
    # torch.randn(n_points*n_points, 2)
], 1)
phi_0 = 0.9*np.pi
x0 = np.array([phi_0, 0])


def test_error(model, x, u, plot, t=0):
    truth = f_star(grid)
    loss_function = nn.MSELoss()
    predictions = model.forward_x(grid.clone()).squeeze()
    # # print(f'prediction {predictions.shape} target {truth.shape} ')
    loss = loss_function(predictions, truth)
    if plot:
        plot_pendulum(x)
        plot_portrait(model.forward_x)
        plt.pause(0.1)
        plt.close()
    # print(f'loss = {loss}')
    return loss


def plot_pendulum(x):
    plt.scatter(x[0], x[1])
    plt.xlim((-phi_max, phi_max))
    plt.ylim((-dphi_max, dphi_max))
    plt.gca().set_aspect('equal', adjustable='box')


def plot_portrait(f):
    predictions = f(grid)

    plt.xlim((-phi_max, phi_max))
    plt.ylim((-dphi_max, dphi_max))
    vector_x = predictions[:, 0].reshape(n_points, n_points).detach().numpy()
    vector_y = predictions[:, 1].reshape(n_points, n_points).detach().numpy()
    magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
    linewidth = magnitude / magnitude.max()
    plt.streamplot(
        grid_q.numpy().T,
        grid_p.numpy().T,
        vector_x.T,
        vector_y.T,
        color='black',
        linewidth=linewidth)
