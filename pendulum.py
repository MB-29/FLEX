import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

import torch 
import torch.nn as nn

from neural_agent import Random, Passive, Periodic, Spacing, OptimalDesign
import environments.pendulum as pendulum

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

plot = False
# plot = True

d, m = 2, 1

period = 2*np.pi / np.sqrt(pendulum.omega_2)
gamma = 0.5
T = 200
dt = 1e-2 * period
sigma = 0.01


class Model(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
    
    def transform(self, z):
        return z[:, :d]

    def forward_x(self, x):
        dx = torch.zeros_like(x)
        dx[:, 0] = x[:, 1]
        # x[:, 1] = torch.sin(x[:, 1])
        dx[:, 1] = self.net(x).view(-1)
        return dx

    def forward_u(self, dx, u):
        dx[:, 1] += u.view(-1)
        return dx
    
    def predictor(self, z):
        zeta = self.transform(z)
        return self.net(zeta).view(-1)


    def forward(self, z):
        x = self.transform(z)
        # x = z[:, :d]
        u = z[:, d]

        dx = self.forward_x(x)
        dx = self.forward_u(dx, u)
        return dx


net = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    # nn.Linear(16, 16),
    # nn.Tanh(),
    nn.Linear(16, 1)
)
model = Model(net)

phi_0 = 0.9*np.pi
x0 = np.array([phi_0, 0])

fig = plt.figure(figsize=(7,4))

plt.subplot(121)
plt.title(r'$f_\theta, \quad t=0$')
pendulum.plot_portrait(model.forward_x)

# agent = Passive(x0.copy(), m, pendulum.dynamics, model, gamma, dt)
# agent = Random(x0.copy(), m, pendulum.dynamics, model, gamma, dt)
# agent = Periodic(x0.copy(), m, pendulum.dynamics, model, gamma, dt)
agent = OptimalDesign(x0.copy(), m, pendulum.dynamics, model, gamma, dt)
# agent = Spacing(x0.copy(), m, pendulum.dynamics, model, gamma, dt)

test_values = agent.identify(T, test_function=pendulum.test_error, plot=plot)
plt.subplot(122)
plt.title(fr'$f_\theta, \quad t={T}$')
pendulum.plot_portrait(model.forward_x)

plt.savefig('learnt-portraits.pdf')
plt.show()


fig = plt.figure(figsize=(7,4))

plt.subplot(223)
plt.plot(agent.x_values[:, 0], agent.x_values[:, 1], alpha=.5, color='black')
plt.title(r'$f_\star$')
pendulum.plot_portrait(pendulum.f_star)
# plt.show()
# plt.savefig('trajectory.pdf')

# plt.savefig('portraits.pdf')
plt.subplot(224)
plt.plot(np.arange(T), test_values, color="black", lw=1)
plt.xlabel(r'$t$')
plt.title(r'Loss')
# plt.savefig('pendulum-loss.pdf')
# plt.yticks([])
# plt.yscale('log')
plt.show()