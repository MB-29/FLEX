from black import Line
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

import torch 
import torch.nn as nn

from neural_agent import Random, Passive, Periodic, Spacing, OptimalDesign, Linearized
import environments.quadrotor as quadrotor

rc('font', size=15)
rc('text', usetex=True)
rc('text.latex', preamble=[r'\usepackage{amsmath}', r'\usepackage{amsfonts}'])

plot = False
# plot = True

d, m = 6, 2

T = 500
dt = 0.1
sigma = 0.01
gamma = quadrotor.gamma



class Model(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def get_B(self, X):
        return dt*quadrotor.get_B(X)
    
    def transform(self, z):
        return z[:, 1:4:2]


    def forward(self, z):
        dx = torch.zeros_like(z[:, :d])
        v = self.transform(z)
        phi = z[:, 4]
        u = z[:, d:]
        c_phi, s_phi, ux, uy = torch.cos(phi), torch.sin(phi), u[:, 0], u[:, 1]
        friction = self.net(v)
        a_x, a_y, a_phi = quadrotor.known_acceleration(c_phi, s_phi, ux, uy)
        dx[:, 0] = z[:, 1]
        dx[:, 2] = z[:, 3]
        dx[:, 4] = z[:, 3]
        dx[:, 1] = a_x
        dx[:, 3] = a_y
        dx[:, 5] = a_phi
        dx[:, 1:4:2] += friction
        return dx
    
    

net = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    # nn.Linear(16, 16),
    # nn.Tanh(),
    nn.Linear(16, 2)
)
# phi0 = np.pi/2
phi0 = 0.1
x0 = quadrotor.X0


model = Model(net)
# agent = Random(x0.copy(), m, quadrotor.dynamics, model, gamma, dt)
# agent = Passive(x0.copy(), m, quadrotor.dynamics, model, gamma, dt)
# agent = Spacing(x0.copy(), m, quadrotor.dynamics, model, gamma, dt)
# agent = OptimalDesign(x0.copy(), m, quadrotor.dynamics, model, gamma, dt)
agent = Linearized(x0.copy(), m, quadrotor.dynamics, model, gamma, dt)
test_values = agent.identify(T, test_function=quadrotor.test_error, plot=plot)

fig = plt.figure(figsize=(14,8))
plt.subplot(211)
plt.plot(agent.x_values[:, 1], agent.x_values[:, 3], alpha=.5, color='black')
plt.subplot(212)
plt.plot(np.arange(T), test_values, color="black", lw=1)
plt.show()