from re import S
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

mass, Mass, l = 1.0, 2.0, 1.0
g = 9.8

alpha, beta = 1.0, 5.0

def acceleration(d_y, c_phi, s_phi, d_phi, u):
    friction_y = -beta * d_y
    friction_phi = - alpha * d_phi
    dd_y = u + mass*l*s_phi*d_phi**2 + friction_y + friction_phi*c_phi/l - mass*g*s_phi*c_phi
    dd_y /=  Mass - mass *c_phi**2

    dd_phi = Mass*g*s_phi - c_phi*(u+mass*l*d_phi**2*s_phi + friction_y) + Mass*friction_phi/(mass*l)
    dd_phi /= Mass*l - mass*l*c_phi**2

    return dd_y, dd_phi

def dynamics(x, u):
    y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    x_dot = np.zeros_like(x)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    dd_y, dd_phi = acceleration(d_y, c_phi, s_phi, d_phi, u)
    # noise = sigma * np.random.randn(d)
    x_dot[0] = d_y
    x_dot[1] = dd_y
    x_dot[2] = d_phi 
    x_dot[3] = dd_phi
    return x_dot

def f_star(z):
    d_y, c_phi, s_phi, d_phi, u = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4]
    dd_y, dd_phi = acceleration(d_y, c_phi, s_phi, d_phi, u)
    dd = torch.zeros_like(z[:, :2])
    # dx[:, 0] = z[:, 1]
    # dx[:, 2] = z[:, 1]
    dd[:, 0] = dd_y
    dd[:, 1] = dd_phi
    return dd


n_points = 10
phi_max = np.pi
dphi_max = np.sqrt(2*g/l)
dy_max = np.sqrt(2*mass*g*l/Mass)
interval_dy = torch.linspace(-dy_max, dy_max, n_points)
interval_phi = torch.linspace(-1, 1, n_points)
interval_dphi = torch.linspace(-dphi_max, dphi_max, n_points)
interval_u = torch.linspace(-10, 10, n_points)
grid_dy, grid_phi, grid_dphi, grid_u = torch.meshgrid(
    interval_dy,
    interval_phi,
    interval_dphi,
    interval_u
    )
grid = torch.cat([
    grid_dy.reshape(-1, 1),
    grid_phi.reshape(-1, 1),
    grid_phi.reshape(-1, 1),
    grid_dphi.reshape(-1, 1),
    grid_u.reshape(-1, 1),
], 1)
# phi_0 = 0.9*np.pi
# x0 = np.array([phi_0, 0])


def test_error(model, x, u, plot, t=0):
    truth = f_star(grid)
    loss_function = nn.MSELoss()
    predictions = model.net(grid.clone()).squeeze()
    # # # print(f'prediction {predictions.shape} target {truth.shape} ')
    loss = loss_function(predictions, truth)
    if plot:
        # plot_cartpole(x)
        plot_portrait(model.net)
        plt.pause(0.1)
        plt.close()
    # print(f'loss = {loss}')
    return loss

def plot_cartpole(x):
    y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    plt.plot([y-l/2, y+l/2], [0, 0], color='black')
    plt.plot([y, y+l*s_phi], [0, l*c_phi], color='red')
    plt.xlim((-2*l, 2*l))
    plt.ylim((-2*l, 2*l))
    plt.gca().set_aspect('equal', adjustable='box')


def plot_portrait(f):
    predictions = f(grid)

    # plt.xlim((-phi_max, phi_max))
    # plt.ylim((-dphi_max, dphi_max))
    vectors = predictions.shape(n_points, n_points, n_points, n_points, 2)
    vector_x = predictions[:, 0].reshape(n_points, n_points).detach().numpy()
    vector_y = predictions[:, 1].reshape(n_points, n_points).detach().numpy()
    magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
    linewidth = magnitude / magnitude.max()
    plt.streamplot(
        grid_dy.numpy().T,
        grid_dphi.numpy().T,
        vector_x.T,
        vector_y.T,
        color='black',
        linewidth=linewidth)