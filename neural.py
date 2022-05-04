from distutils.log import error
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

from neural_agent import Agent

d, m, q = 4, 1, 3

mass, Mass, l = 0.9, 1.5, 2.5
g = 10

gamma = 0.01
dt = 1e-1
T = 100
sigma = 0.01

net = nn.Sequential(
    nn.Linear(d+m, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, d)
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
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    dx[0] = d_y
    dx[1] = (u + mass*s_phi*(l*d_phi**2 - g*c_phi) ) / (Mass + mass*s_phi**2)
    # dx[2] = -d_phi * s_phi
    dx[2] = d_phi
    dx[3] = -mass*l*c_phi*s_phi*d_phi**2  -u*c_phi + (mass+Mass)*g*s_phi 
    dx[3] /= l*(Mass+mass*s_phi**2)
    print(f'x = {x}, dx={dx}, u ={u}')
    dx *= dt
    plot_cartpole(x)
    plt.pause(0.1)
    plt.close()
    return x + dx

def f_star(z):
    x = z[:, :d]
    u = z[:, d:].squeeze()
    y, d_y, phi, d_phi = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    c_phi, s_phi = torch.cos(phi), torch.sin(phi)
    dx = torch.zeros_like(x)
    dx[:, 0] = d_y
    dx[:, 1] = (u + mass*s_phi*(l*d_phi**2 - g*c_phi) ) / (Mass + mass*s_phi**2)
    # dx[:, 2] = -d_phi * s_phi
    dx[:, 2] = d_phi
    dx[:, 3] = -mass*l*c_phi*s_phi*d_phi**2  -u*c_phi + (mass+Mass)*g*s_phi 
    dx[:, 3] /= l*(Mass+mass*s_phi**2)
    dx *= dt
    return x +  dx

def noisy_dynamics(x, u):
    noise = sigma * np.random.randn(d)
    return dynamics(x,u) + noise


def test_error(net, batch_size=32):
    Z = 0.1*torch.randn(batch_size, d+m)
    Z[:, :d] += x0
    predictions = net(Z)
    truth = f_star(Z)
    loss_function = nn.MSELoss()
    return loss_function(predictions, truth)

phi0 = 0.9*np.pi
x0 = np.array([0, 0, phi0,  0])

# theta0 = 2*np.random.rand(q)
agent = Agent(x0, m, dynamics, net, gamma)

test_values = agent.identify(T, test_function=test_error)

plt.plot(test_values)
plt.show()