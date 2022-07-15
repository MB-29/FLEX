import numpy as np
import torch
import torch.nn as nn

import environments.pendulum as pendulum

d, m = pendulum.d, pendulum.m
dt = pendulum.dt


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def get_B(self, x):
        return dt*np.eye(d, m)

    def transform(self, z):
        return z[:, :d]

    def forward_x(self, x):
        raise NotImplementedError

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

class NeuralModel(Model):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            # nn.Linear(16, 16),
            # nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward_x(self, x):
        dx = torch.zeros_like(x)
        dx[:, 0] = x[:, 1]
        # x[:, 1] = torch.sin(x[:, 1])
        dx[:, 1] = self.net(x).view(-1)
        return dx

class LinearModel(Model):
    def __init__(self):
        super().__init__()
        self.theta = nn.parameter.Parameter(torch.zeros(2, dtype=torch.float))
        self.lr = 0.01
    
    def forward_x(self, x):
        dx = torch.zeros_like(x)
        dx[:, 0] = x[:, 1]
        dx[:, 1] = self.theta[0] * torch.sin(x[:, 0]) + self.theta[1]*x[:, 1]
        return dx
        