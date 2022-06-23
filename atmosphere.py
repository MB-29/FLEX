from distutils.log import error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

import torch 
import torch.nn as nn

from neural_agent import Random, Passive, Spacing
import environments.atmosphere as atmosphere

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

plot = False
plot = True

d, m = 4, 2


gamma = 0.1
R = 1
dt = 0.2*1e-2 * atmosphere.period
T = 1000
sigma = 0.01

class Model(nn.Module):

    def __init__(self, direction_net):
        super().__init__()
        # self.magnitude_net = magnitude_net
        self.direction_net = direction_net
    
    def transform(self, z):
        return z[:, :d]

    def prediction(self, position):
        r = torch.sqrt(position[:, 0]**2 + position[:, 1]**2).unsqueeze(1)
        # magnitude = self.magnitude_net(r)
        direction = self.direction_net(position)
        # direction = direction/torch.linalg.norm(direction, dim=1).unsqueeze(1)
        acceleration = direction
        return acceleration


    def forward(self, z):
        x = z[:, :d]
        u = z[:, d:]
        dx = torch.zeros_like(x)
        dx[:, ::2] = x[:, 1::2]
        position = x[:, 0::2]
        acceleration = self.prediction(position)
        acceleration += u - atmosphere.alpha * x[:, 1::2]
        dx[:, 1::2] = acceleration
        return dx

magnitude_net = nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),
    # nn.Linear(16, 16),
    # nn.Tanh(),
    # nn.Linear(8, 8),
    # nn.Tanh(),
    nn.Linear(16, 1)
)
direction_net = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    # nn.Linear(16, 16),
    # nn.Tanh(),
    # nn.Linear(8, 8),
    # nn.Tanh(),
    nn.Linear(16, 2)
)
model = Model(direction_net)
x0 = np.array([atmosphere.R, 0, 0,  atmosphere.R*atmosphere.omega])


agent = Passive(x0, m, atmosphere.dynamics, model, gamma, dt)
# agent = Random(x0, m, atmosphere.dynamics, model, gamma, dt)
agent = Spacing(x0, m, atmosphere.dynamics, model, gamma, dt)

test_values = agent.identify(T, test_function=atmosphere.test_error, plot=plot)

plt.plot(test_values, color="black", lw=1)
plt.xlabel(r'$t$')
plt.title('loss')
# plt.yticks([])
# plt.yscale('log')
# plt.savefig('atmosphere-loss.pdf')
plt.show()