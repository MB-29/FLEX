from distutils.log import error
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

from neural_agent import Random

d, m, q = 4, 1, 3

mass, Mass, l = 1.0, 2.0, 1.0
g = 10
mu, kappa = 5, 1

gamma = 1
dt = 1e-2 * 2*np.pi*np.sqrt(l/g)
T = 5000
sigma = 0.01

net = nn.Sequential(
    nn.Linear(d+m, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 8),
    nn.Tanh(),
    nn.Linear(8, d)
)

def plot_cartpole(x):
    y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    plt.plot([y-l/2, y+l/2], [0, 0], color='black')
    plt.plot([y, y+l*s_phi], [0, +l*c_phi], color='red')
    plt.xlim((-2*l, 2*l))
    plt.ylim((-2*l, 2*l))
    plt.gca().set_aspect('equal', adjustable='box')


def dynamics(x, u):
    y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    dx = np.zeros_like(x)
    x[2] %= 2*np.pi
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    dx[0] = d_y
    dx[1] = (u + mass*s_phi*(l*d_phi**2 - g*c_phi) - mu*d_y ) / (Mass + mass*s_phi**2)
    # dx[2] = -d_phi * s_phi
    dx[2] = d_phi
    dx[3] = -mass*l*c_phi*s_phi*d_phi**2  -u*c_phi + (mass+Mass)*g*s_phi + mu*d_y*c_phi
    dx[3] /= l*(Mass+mass*s_phi**2)
    # print(f'x = {x}, dx={dx}, u ={u}')
    # plot_cartpole(x)
    # plt.pause(0.1)
    # plt.close()
    return  dx

def f_star(z):
    x = z[:, :d]
    x[:, 2] %= 2*np.pi
    u = z[:, d:].squeeze()
    y, d_y, phi, d_phi = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    phi = phi % (2*np.pi)
    c_phi, s_phi = torch.cos(phi), torch.sin(phi)
    dx = torch.zeros_like(x)
    dx[:, 0] = d_y
    dx[:, 1] = (u + mass*s_phi*(l*d_phi**2 - g*c_phi) -mu*d_y ) / (Mass + mass*s_phi**2)
    # dx[:, 2] = -d_phi * s_phi
    dx[:, 2] = d_phi
    dx[:, 3] = -mass*l*c_phi*s_phi*d_phi**2  -u*c_phi + (mass+Mass)*g*s_phi + mu*d_y*c_phi
    dx[:, 3] /= l*(Mass+mass*s_phi**2)
    return   dx

def noisy_dynamics(x, u):
    noise = sigma * np.random.randn(d)
    return dynamics(x,u) + noise


phi0 = 0.9*np.pi
x0 = np.array([0, 0, phi0,  0])
Z = torch.randn(1000, d+m)
Z[:, :d] += x0
Z[:, 2] %= 2*np.pi
def test_error(net):
    predictions = net(Z)
    truth = f_star(Z)
    loss_function = nn.MSELoss()
    return loss_function(predictions, truth)


# theta0 = 2*np.random.rand(q)
agent = Random(x0, m, dynamics, net, gamma, dt)

test_values = agent.identify(T, test_function=test_error)

plt.plot(test_values)
# plt.yscale('log')
plt.show()