import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

d, m = 4, 2

alpha = 0.1
m1, m2 = 1, 1
l1, l2 = 1, 1
g = 9.8

period = 2*np.pi * np.sqrt(l1 / g)
T = 1000
dt = 1e-2 * period
sigma = 0.01

gamma = 1

x0 = np.array([0.0, 0.0, 0.0, 0.0])


def get_B(x):
    phi1, d_phi1, phi2, d_phi2 = x[0], x[1], x[2], x[3]
    c_phi1, s_phi1 = np.cos(phi1), np.sin(phi1)
    c_phi2, s_phi2 = np.cos(phi2), np.sin(phi2)
    c_dphi = np.cos(phi1-phi2)
    s_dphi = np.sin(phi1-phi2)
    M = (m1 + m2 * s_dphi**2)
    B = np.array([
        [0, 0],
        [1 / (l1*M), -c_dphi / (l1*M)],
        [0, 0],
        [-c_dphi / (l2*M), ((m1+m2)/m2) / (l2*M)]
    ])
    return B

def acceleration_x(c_phi1, c_phi2, s_phi1, s_phi2, c_dphi, s_dphi, s_2dphi, d_phi1, d_phi2):
    friction1 = - alpha*d_phi1
    friction2 = - alpha*d_phi2

    a_phi1 = m2*l1*d_phi1**2*s_2dphi
    a_phi1 += 2*m2*l2*d_phi2**2*s_dphi
    a_phi1 += 2*g*m2*c_phi2*s_dphi
    a_phi1 += 2*g*m1*s_phi1
    a_phi1 += -2*friction1 + 2*friction2*c_dphi
    a_phi1 /= -2*l1*(m1+m2*s_dphi**2)

    a_phi2 = m2*l2*d_phi2**2*s_2dphi
    a_phi2 += 2*(m1+m2)*l1*d_phi1**2*s_dphi
    a_phi2 += 2*g*(m1+m2)*c_phi1*s_dphi
    a_phi2 += 2*((m1+m2)/m2)*friction2 - 2*friction1*c_dphi
    a_phi2 /= 2*l2*(m1+m2*s_dphi**2)

    return a_phi1, a_phi2
    
def acceleration_u(u1, u2, c_dphi, s_dphi):
    a_phi1 = -2*u1
    a_phi1 /= -2*l1*(m1+m2*s_dphi**2)
    a_phi2 = 2*((m1+m2)/m2)*u2 - 2*u1*c_dphi
    a_phi2 /= 2*l2*(m1+m2*s_dphi**2)

    return a_phi1, a_phi2

def dynamics(x, u):
    assert np.linalg.norm(u) < 1.1*gamma
    phi1, d_phi1, phi2, d_phi2 = x[0], x[1], x[2], x[3]
    c_phi1, c_phi2 = np.cos(phi1), np.cos(phi2)
    s_phi1, s_phi2 = np.sin(phi1), np.sin(phi2)
    dphi = phi1 - phi2
    c_dphi, s_dphi = np.cos(dphi), np.sin(dphi)
    s_2dphi = np.sin(2*(dphi))
    dx = np.zeros_like(x)
    dx[0] = x[1]
    dx[2] = x[3]
    a_phi1, a_phi2 = acceleration_x(
        c_phi1, c_phi2, s_phi1, s_phi2, c_dphi, s_dphi, s_2dphi, d_phi1, d_phi2)
    a_phi1_u, a_phi2_u = acceleration_u(u[0], u[1], c_dphi, s_dphi)
    dx[1] = a_phi1 + a_phi1_u
    dx[3] = a_phi2 + a_phi2_u
    return dx


def f_star(zeta):
    c_phi1, c_phi2, s_phi1, s_phi2 = zeta[:, 0], zeta[:, 1], zeta[:, 2], zeta[:, 3]
    d_phi1, d_phi2 = zeta[:, 4], zeta[:, 5]
    c_dphi = c_phi1*c_phi2 + s_phi1*s_phi2
    s_dphi = s_phi1*c_phi2 - c_phi1*s_phi2
    s_2dphi = 2*s_dphi*c_dphi
    dx = torch.zeros_like(zeta[:, :2])
    a_phi1, a_phi2 = acceleration_x(
        c_phi1, c_phi2, s_phi1, s_phi2, c_dphi, s_dphi, s_2dphi, d_phi1, d_phi2)
    dx[:, 0] = a_phi1
    dx[:, 1] = a_phi2
    return dx


n_points = 10
phi_max = np.pi
dphi_max = 2*np.sqrt(2*g/(l1+l2))
interval_phi = torch.linspace(0, 2*np.pi, n_points)
interval_dphi = torch.linspace(-dphi_max, dphi_max, n_points)
grid_phi1, grid_phi2, grid_d_phi1, grid_d_phi2 = torch.meshgrid(
    interval_phi,
    interval_phi,
    interval_dphi,
    interval_dphi,
)
grid = torch.cat([
    torch.cos(grid_phi1.reshape(-1, 1)),
    torch.cos(grid_phi2.reshape(-1, 1)),
    torch.sin(grid_phi1.reshape(-1, 1)),
    torch.sin(grid_phi2.reshape(-1, 1)),
    grid_d_phi1.reshape(-1, 1),
    grid_d_phi2.reshape(-1, 1)
], 1)


def test_error(model, x, u, plot, t=0):
    truth = f_star(grid)
    loss_function = nn.MSELoss()
    predictions = model.net(grid.clone()).squeeze()
    # # # print(f'prediction {predictions.shape} target {truth.shape} ')
    loss = loss_function(predictions, truth)
    if plot and t % 10 == 0:
        plot_arm(x, u)
        plt.title(t)
        # plot_portrait(model.net)
        plt.pause(0.1)
        plt.close()
    # print(f'loss = {loss}')
    return loss


def plot_arm(x, u):
    phi1, d_phi1, phi2, d_phi2 = x[0], x[1], x[2], x[3]
    # push = 0.7*np.sign(np.mean(u))
    c_phi1, s_phi1 = np.cos(phi1), np.sin(phi1)
    c_phi2, s_phi2 = np.cos(phi2), np.sin(phi2)
    # plt.arrow(y, 0, push, 0, color='red', head_width=0.1, alpha=0.5)
    plt.plot([0, l1*s_phi1], [0, -l1*c_phi1], color='black')
    plt.plot([l1*s_phi1, l1*s_phi1 + l2*s_phi2],
             [-l1*c_phi1, -l1*c_phi1-l2*c_phi2], color='blue')
    plt.xlim((-(l1+l2), l1+l2))
    plt.ylim((-(l1+l2), l1+l2))
    plt.gca().set_aspect('equal', adjustable='box')
