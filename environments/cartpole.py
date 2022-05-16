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


def f_star(x):
    y, d_y, phi, d_phi = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    c_phi, s_phi = torch.cos(phi), torch.sin(phi)
    dd_y, dd_phi = acceleration(d_y, c_phi, s_phi, d_phi)
    x_dot = torch.zeros_like(x)
    x_dot[:, 0] = d_y
    x_dot[:, 1] = dd_y
    x_dot[:, 2] = d_phi
    x_dot[:, 3] = dd_phi
    return x_dot


# n_points = 100
# phi_max = np.pi
# dphi_max = np.sqrt(-2*omega_2*np.cos(phi_max))
# interval_q = torch.linspace(-phi_max, phi_max, n_points)
# interval_p = torch.linspace(-dphi_max, dphi_max, n_points)
# grid_q, grid_p = torch.meshgrid(interval_q, interval_p)
# grid = torch.cat([
#     grid_q.reshape(-1, 1),
#     grid_p.reshape(-1, 1),
#     # torch.randn(n_points*n_points, 2)
# ], 1)
# phi_0 = 0.9*np.pi
# x0 = np.array([phi_0, 0])


def test_error(model, x, u, plot, t=0):
    # truth = f_star(grid)
    # loss_function = nn.MSELoss()
    # predictions = model.forward_x(grid.clone()).squeeze()
    # # # print(f'prediction {predictions.shape} target {truth.shape} ')
    # loss = loss_function(predictions, truth)
    loss = torch.zeros(1)
    if plot:
        plot_cartpole(x)
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

