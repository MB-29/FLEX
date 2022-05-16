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
gamma = 1
T = 200
dt = 1e-2 * period
sigma = 0.01


class Model(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward_x(self, x):
        dx = torch.zeros_like(x)
        dx[:, 0] = x[:, 1]
        dx[:, 2] = x[:, 3]
        # x[:, 1] = torch.sin(x[:, 1])
        dx[:, 1::2] = self.net(x).view(-1)
        return dx

    def forward_u(self, dx, u):
        dx[:, 1] += u.view(-1)
        dx[:, 1] += u.view(-1)
        return dx

    def forward(self, z):
        x = z[:, :d]
        u = z[:, d]

        dx = self.forward_x(x)
        dx = self.forward_u(dx, u)
        return dx


net = nn.Sequential(
    nn.Linear(4, 16),
    nn.Tanh(),
    # nn.Linear(16, 16),
    # nn.Tanh(),
    nn.Linear(16, 2)
)
model = Model(net)



x0 = np.array([0.0, 0.0, np.pi/2, 0.0])

fig = plt.figure(figsize=(14,8))


agent = Random(x0.copy(), m, cartpole.dynamics, model, gamma, dt)

test_values = agent.identify(T, test_function=cartpole.test_error, plot=True)
plt.subplot(211)
plt.scatter(agent.x_values[:, 2], agent.x_values[:, 3], alpha=.5, marker='x', color='black')
plt.subplot(212)
plt.plot(np.arange(T), test_values, color="black", lw=1)
plt.show()