import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

d, m = 2, 1
g = 10.0

l = 1.0
alpha = 0
omega_2 = (3/2)*(g /l)

period = 2*np.pi / np.sqrt(omega_2)
gamma = 2
T = 2000
dt = 80e-4
sigma = 0

A_star = torch.tensor([
    [0, 0, 1],
    [0, -omega_2, 0]
])

def dynamics(x, u):
    dx = np.zeros_like(x)
    dx[0] = x[1]
    dx[1] = -omega_2 * np.sin(x[0]) + u
    noise = sigma * np.random.randn(d)
    dx += noise
    return dx


def f_star(zeta):
    dx = torch.zeros_like(zeta[:, :d])
    dx[:, 0] = zeta[:, 2]
    dx[:, 1] = -omega_2 * zeta[:, 1]
    return dx


n_points = 100
phi_max = np.pi
dphi_max = np.sqrt(-2*omega_2*np.cos(phi_max))
interval_phi = torch.linspace(-phi_max, phi_max, n_points)
interval_dphi = torch.linspace(-dphi_max, dphi_max, n_points)
grid_phi, grid_dphi = torch.meshgrid(interval_phi, interval_dphi)
grid = torch.cat([
    torch.cos(grid_phi.reshape(-1, 1)),
    torch.sin(grid_phi.reshape(-1, 1)),
    grid_dphi.reshape(-1, 1)
], 1)

x0 = np.array([0.0, 0.0])


def test_error(model, x, u, plot, t=0):
    truth = f_star(grid)
    loss_function = nn.MSELoss()
    predictions = model.a_net(grid.clone()).squeeze()
    # # print(f'prediction {predictions.shape} target {truth.shape} ')
    loss = loss_function(predictions, truth)
    # loss = torch.linalg.norm(A_star-model.a_net[0].weight)
    if plot and t%5 == 0:
        plot_pendulum(x)
        # plot_portrait(model.forward_x)
        plt.pause(0.1)
        plt.close()
    # print(f'loss = {loss}')
    # print(x)
    return loss


def plot_phase(x):
    plt.scatter(x[0], x[1])
    plt.xlim((-phi_max, phi_max))
    plt.ylim((-dphi_max, dphi_max))
    plt.gca().set_aspect('equal', adjustable='box')


def plot_pendulum(x):
    phi = x[0]
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    plt.plot([0, l*s_phi], [0, -l*c_phi], color='blue')
    plt.xlim((-2*l, 2*l))
    plt.ylim((-2*l, 2*l))
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
        grid_phi.numpy().T,
        grid_dphi.numpy().T,
        vector_x.T,
        vector_y.T,
        color='black',
        linewidth=linewidth)
    # plt.gca().set_aspect('equal', adjustable='box')
