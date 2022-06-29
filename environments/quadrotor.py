from re import S
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

m, I, r = 1.0, 10, 1.0
g = 0
rho = 0.2   

gamma = 10

def get_B(X):
    x, v_x, y, v_y, phi, d_phi = X[0], X[1], X[2], X[3], X[4], X[5]
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    B = np.array([
        [0, 0],
        [-s_phi / m, -s_phi / m],
        [0, 0],
        [c_phi / m, c_phi / m],
        [0, 0],
        [r/I, -r/I]
    ])
    return B

def known_acceleration(c_phi, s_phi, u1, u2):
    a_x = -s_phi*(u1 + u2) / m
    a_y = (c_phi*(u1 + u2)  - m*g)/m
    a_phi = r*(u1 - u2) / I
    return a_x, a_y, a_phi


def dynamics(X, u):
    magnitude = np.linalg.norm(u)
    # print(f'magnitude= {magnitude}')
    x, v_x, y, v_y, phi, d_phi = X[0], X[1], X[2], X[3], X[4], X[5]
    X_dot = np.zeros_like(X)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    f_x = - (rho/m) * np.abs(v_x) * v_x
    f_y = - (rho/m) * np.abs(v_y) * v_y
    a_x, a_y, a_phi = known_acceleration(c_phi, s_phi, u[0], u[1])
    # print(f'phi = {phi%(2*np.pi)}')
    # print(f'a_x = {a_x}')
    X_dot[0] = X[1]
    X_dot[1] = a_x + f_x
    X_dot[2] = X[3] 
    X_dot[3] = a_y + f_y
    X_dot[4] = X[5]
    X_dot[5] = a_phi
    return X_dot

def g_star(v):
    return -(rho/m)*torch.abs(v)*v

y0 = 0.0
X0 = np.array([0, 0, y0, 0, 0, 0])
n_points = 20
v_max = 2*np.sqrt(gamma)
interval_v = torch.linspace(-v_max, v_max, n_points)
grid_vx, grid_vy = torch.meshgrid(
    interval_v,
    interval_v,
    )
grid = torch.cat([
    grid_vy.reshape(-1, 1),
    grid_vy.reshape(-1, 1)
    # grid_u.reshape(-1, 1),
], 1)
# phi_0 = 0.9*np.pi
# x0 = np.array([phi_0, 0])

def test_error(model, X, u, plot, t=0):
    truth = g_star(grid)
    loss_function = nn.MSELoss()
    predictions = model.net(grid.clone()).squeeze()
    # # # print(f'prediction {predictions.shape} target {truth.shape} ')
    loss = loss_function(predictions, truth)
    if plot and t%10==0:
        plt.subplot(121)
        plot_quadrotor(X, u)
        plt.subplot(122)
        # plot_portrait(g_star)
        plot_portrait(model.net)
        plt.pause(0.1)
        plt.close()
    # print(f'loss = {loss}')
    return loss

def plot_quadrotor(X, u):
    x, v_x, y, v_y, phi, d_phi = X[0], X[1], X[2], X[3], X[4], X[5]
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    l0 = u[0]/gamma
    l1 = u[1]/gamma
    plt.arrow(x-r*c_phi, y-r*s_phi, -l0*s_phi, l0*c_phi,
              color='red', head_width=0.1, alpha=0.5)
    plt.arrow(x+r*c_phi, y+r*s_phi, -l1*s_phi, l1*c_phi,
              color='red', head_width=0.1, alpha=0.5)
    plt.plot([x-r*c_phi, x+r*c_phi], [y-r*s_phi, y+r*s_phi], color='blue')
    # plt.xlim((x-2*r, x+2*r))
    # plt.ylim((y-2*r, y+2*r))
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
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
        grid_vx.numpy().T,
        grid_vy.numpy().T,
        vector_x.T,
        vector_y.T,
        color='black',
        linewidth=linewidth
        )
