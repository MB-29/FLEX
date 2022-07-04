from re import S
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

d, m = 4, 1

mass, Mass, l = 1.0, 2.0, 1.0
g = 9.8
alpha, beta = 0, 5.0
phi0 = 0.1
x0 = np.array([0.0, 0.0, phi0, 0.0])

period = 2*np.pi * np.sqrt(l / g)
T = 1000
dt = 1e-2 * period
sigma = 0.01

gamma = 2

def get_B(x):
    y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    B = np.array([
        [0],
        [1/(Mass - mass * c_phi**2)],
        [0],
        [- c_phi/ (Mass*l - mass*l*c_phi**2)]
    ])
    return B

def acceleration_x(d_y, c_phi, s_phi, d_phi):
    friction_y = -beta * d_y
    friction_phi = - alpha * d_phi
    dd_y =  mass*l*s_phi*d_phi**2 + friction_y + friction_phi*c_phi/l - mass*g*s_phi*c_phi
    dd_y /=  Mass - mass *c_phi**2

    dd_phi = Mass*g*s_phi - c_phi*(mass*l*d_phi**2*s_phi + friction_y) + Mass*friction_phi/(mass*l)
    dd_phi /= Mass*l - mass*l*c_phi**2

    return dd_y, dd_phi
    
def acceleration_u(c_phi, u):
    dd_y = u / (Mass - mass *c_phi**2)
    dd_phi =  - c_phi*u/ (Mass*l - mass*l*c_phi**2)

    return dd_y, dd_phi

def dynamics(x, u):
    y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    x_dot = np.zeros_like(x)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    dd_y, dd_phi = acceleration_x(d_y, c_phi, s_phi, d_phi)
    dd_y_u, dd_phi_u = acceleration_u(c_phi, u)
    dd_y += dd_y_u
    dd_phi += dd_phi_u
    # noise = sigma * np.random.randn(d)
    x_dot[0] = d_y
    x_dot[1] = dd_y
    x_dot[2] = d_phi 
    x_dot[3] = dd_phi
    return x_dot

def f_star(z):
    d_y, c_phi, s_phi, d_phi = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    dd_y, dd_phi = acceleration_x(d_y, c_phi, s_phi, d_phi)
    dd = torch.zeros_like(z[:, :2])
    # dx[:, 0] = z[:, 1]
    # dx[:, 2] = z[:, 1]
    dd[:, 0] = dd_y
    dd[:, 1] = dd_phi
    return dd


n_points = 20
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
    grid_dphi.reshape(-1, 1)
    # grid_u.reshape(-1, 1),
], 1)

def test_error(model, x, u, plot, t=0):
    truth = f_star(grid)
    loss_function = nn.MSELoss()
    predictions = model.net(grid.clone()).squeeze()
    # # # print(f'prediction {predictions.shape} target {truth.shape} ')
    loss = loss_function(predictions, truth)
    if plot and t%10==0:
        plot_cartpole(x, u)
        # plot_portrait(model.net)
        plt.pause(0.1)
        plt.close()
    # print(f'loss = {loss}')
    return loss

def plot_cartpole(x, u):
    y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    push = 0.7*np.sign(np.mean(u))
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    plt.arrow(y, 0, push, 0, color='red', head_width=0.1, alpha=0.5)
    plt.plot([y-l/2, y+l/2], [0, 0], color='black')
    plt.plot([y, y+l*s_phi], [0, l*c_phi], color='blue')
    plt.xlim((y-2*l, y+2*l))
    plt.ylim((-2*l, 2*l))
    plt.gca().set_aspect('equal', adjustable='box')


def plot_portrait(f):
    predictions = f(grid)

    # plt.xlim((-phi_max, phi_max))
    # plt.ylim((-dphi_max, dphi_max))
    # vectors = predictions.shape(n_points, n_points, n_points, n_points, 2)
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
