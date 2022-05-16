from curses.ascii import SP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from tqdm import tqdm
import torch
import torch.nn as nn

from neural_agent import Random, Passive, OptimalDesign, Spacing

import environments.pendulum as pendulum

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

d, m = 2, 1
g = 1

period = 2*np.pi / np.sqrt(pendulum.omega_2)
gamma = 1
T = 200
dt = 1e-2 * period
sigma = 0.01


class Model(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def transform(self, z):
        return z

    def forward_x(self, x):
        dx = torch.zeros_like(x)
        dx[:, 0] = x[:, 1]
        # x[:, 1] = torch.sin(x[:, 1])
        dx[:, 1] = self.net(x).view(-1)
        return dx

    def forward_u(self, dx, u):
        dx[:, 1] += u.view(-1)
        return dx

    def forward(self, z):
        z = self.transform(z)
        x = z[:, :d]
        u = z[:, d]

        dx = self.forward_x(x)
        dx = self.forward_u(dx, u)
        return dx


phi_0 = 0.9*np.pi
x0 = np.array([phi_0, 0])
# theta0 = 2*np.random.rand(q)
n_samples = 50
# for agent_ in [Random, Active]:
for agent_ in [Random, Passive, OptimalDesign, Spacing]:
# for agent_ in [Spacing]:
    test_values = np.zeros((n_samples, T))
    for sample_index in tqdm(range(n_samples)):
        net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        model = Model(net)
        agent = agent_(x0.copy(), m, pendulum.dynamics, model, gamma, dt)

        test_values[sample_index, :] = agent.identify(
            T, test_function=pendulum.test_error)

    test_mean = np.mean(test_values, axis=0)
    test_yerr = 2 * np.sqrt(np.var(test_values, axis=0) / n_samples)

    plt.plot(test_mean, label=agent, alpha=0.7)
    plt.fill_between(np.arange(T), test_mean-test_yerr,
                     test_mean+test_yerr, alpha=0.5)
plt.legend()
plt.show()
