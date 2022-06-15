import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

import torch 
import torch.nn as nn

from neural_agent import Random, Passive, Oracle, Periodic, Spacing, OptimalDesign
import environments.cartpole as cartpole

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

plot = False
plot = True

d, m = 4, 1

period = 2*np.pi * np.sqrt(cartpole.l / cartpole.g)
gamma = 10
T = 1000
dt = 1e-2 * period
sigma = 0.01


class Model(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
    
    def transform(self, z):
        z_ = z[:, :d].clone()
        z_[:, 0] = z[:, 1]
        phi = z[:, 2]
        z_[:, 1] = torch.cos(phi)
        z_[:, 2] = torch.sin(phi)
        return z_

    def acceleration_u(self, z):
        phi = z[:, 2]
        u = z[:, -1]
        c_phi = torch.cos(phi)
        dd_y_u, dd_phi_u = cartpole.acceleration_u(c_phi, u)
        acceleration = torch.zeros_like(z[:, :2])
        acceleration[:, 0] = dd_y_u
        acceleration[:, 1] = dd_phi_u
        return acceleration

    def forward(self, z):
        dx = torch.zeros_like(z[:, :d])
        x = self.transform(z)
        # x = z[:, :d]
        # u = z[:, d]
        acceleration_x = self.net(x)
        acceleration_u = self.acceleration_u(z)
        acceleration = acceleration_x + acceleration_u
        dx[:, 1::2] = acceleration
        dx[:, 0] = z[:, 1]
        dx[:, 2] = z[:, 3]

        # dx = self.forward_x(x)
        # dx = self.forward_u(dx, u)
        return dx

net = nn.Sequential(
    nn.Linear(4, 16),
    nn.Tanh(),
    # nn.Linear(16, 16),
    # nn.Tanh(),
    nn.Linear(16, 2)
)
# phi0 = np.pi/2
phi0 = 0.1
x0 = np.array([0.0, 0.0, phi0, 0.0])


model = Model(net)
# agent = Random(x0.copy(), m, cartpole.dynamics, model, gamma, dt)
# agent = Passive(x0.copy(), m, cartpole.dynamics, model, gamma, dt)
agent = Spacing(x0.copy(), m, cartpole.dynamics, model, gamma, dt)
# agent = Periodic(x0.copy(), m, cartpole.dynamics, model, gamma, dt)
test_values = agent.identify(T, test_function=cartpole.test_error, plot=plot)

fig = plt.figure(figsize=(14,8))
plt.subplot(211)
plt.plot(agent.x_values[:, 2], agent.x_values[:, 3], alpha=.5, color='black')
plt.subplot(212)
plt.plot(np.arange(T), test_values, color="black", lw=1)
plt.show()