import torch
import torch.nn as nn

import environments.linear as linear

d, m = linear.d, linear.m
dt = linear.dt

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        net = nn.Linear(2, 2, bias=False)

        # net = nn.Sequential(
        #     nn.Linear(2, 16),
        #     nn.Tanh(),
        #     # nn.Linear(16, 16),
        #     # nn.Tanh(),
        #     nn.Linear(16, 2)
        # )
        self.net = net

    def get_B(self, x):
        return dt*linear.B

    def transform(self, z):
        return z[:, :d]

    def forward_x(self, x):
        dx = self.net(x)
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
