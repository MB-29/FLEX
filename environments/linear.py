import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

d, m = 2, 1

gamma = 1
T = 1000
dt = 1 / (np.sqrt(gamma) * T)
sigma = 0.01


A = np.array([
    [0, 1],
    [0, -0.1]
])
B = np.array([
    [0],
    [1]
])
A_ = torch.tensor(A, dtype=torch.float)

def dynamics(x, u):
    dx = A@x + B@u
    return dx


def f_star(x):
    dx = A_@x.T
    return dx.T


n_points = 100
interval = torch.linspace(-1, 1, n_points)
grid_q, grid_p = torch.meshgrid(interval, interval)
grid = torch.cat([
    grid_q.reshape(-1, 1),
    grid_p.reshape(-1, 1),
], 1)


def test_error(model, x, u, plot, t=0):
    truth = f_star(grid)
    loss_function = nn.MSELoss()
    predictions = model.forward_x(grid.clone()).squeeze()
    # # print(f'prediction {predictions.shape} target {truth.shape} ')
    loss = loss_function(predictions, truth)
    if plot and t % 5 == 0:
        plot_point(x)
        plot_portrait(model.forward_x)
        plt.pause(0.1)
        plt.close()
    # print(f'loss = {loss}')
    # print(x)
    return loss


def plot_point(x):
    plt.scatter(x[0], x[1])
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.gca().set_aspect('equal', adjustable='box')


def plot_portrait(f):
    predictions = f(grid)

    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
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
    # plt.gca().set_aspect('equal', adjustable='box')
