import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

mass, Mass, l = 0.9, 1.5, 2.5
g = 10

def dynamics(x, u):
    z, d_z, phi, d_phi = x[0], x[1], x[2], x[3]
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    x_dot = np.zeros_like(x)
    # noise = sigma * np.random.randn(d)
    x_dot[0] = d_z
    x_dot[1] = -mass*l*s_phi*d_phi**2 + u + mass*g*c_phi*s_phi
    x_dot[1] /= Mass + mass*(1-c_phi**2)
    x_dot[2] = d_phi 
    x_dot[3] = -mass*l*c_phi*s_phi*d_phi**2 + \
        u*c_phi + mass*g*s_phi + Mass*g*s_phi
    x_dot[3] /= l*(Mass+mass*(1-c_phi**2))
    return x_dot


def f_star(x):
    z, d_z, phi, d_phi = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    c_phi, s_phi = torch.cos(phi), torch.sin(phi)
    x_dot = torch.zeros_like(x)
    x_dot[:, 0] = d_z
    x_dot[:, 1] = -mass*l*s_phi*d_phi**2  + mass*g*c_phi*s_phi
    x_dot[:, 1] /= Mass + mass*(1-c_phi**2)
    x_dot[:, 1] = d_phi
    x_dot[:, 3] = -mass*l*c_phi*s_phi*d_phi**2 + c_phi + mass*g*s_phi + Mass*g*s_phi
    x_dot[:, 3] /= l*(Mass+mass*(1-c_phi**2))
    return x_dot


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


def test_error(model, x, u, t=0):
    truth = f_star(grid)
    loss_function = nn.MSELoss()
    predictions = model.forward_x(grid.clone()).squeeze()
    # # print(f'prediction {predictions.shape} target {truth.shape} ')
    loss = loss_function(predictions, truth)

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
