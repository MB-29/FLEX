import numpy as np
import torch
import torch.nn as nn

import environments.pendulum_gym as pendulum_gym

d, m = pendulum_gym.d, pendulum_gym.m
dt = pendulum_gym.dt


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def transform(self, z):
        zeta = torch.zeros_like(z)
        zeta[:, 0] = torch.cos(z[:, 0])
        zeta[:, 1] = torch.sin(z[:, 0])
        zeta[:, 2] = z[:, 1]
        return zeta

    def forward_x(self, x):
        raise NotImplementedError

    def forward_u(self, dx, u):
        raise NotImplementedError

    # def predictor(self, z):
        # zeta = self.transform(z)
        # return self.net(zeta).view(-1)


class NeuralModel(Model):

    def __init__(self):
        super().__init__()
        # self.a_net = nn.Sequential(
        #     nn.Linear(3, 2, bias=False),
        # )
        # self.lr = 0.001
    
        self.a_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            # nn.Linear(16, 16),
            # nn.Tanh(),
            nn.Linear(16, 2)
        )
        self.lr = 0.0005
    
        # self.B_net = nn.Sequential(
        #     nn.Linear(3, 16),
        #     nn.Tanh(),
        #     # nn.Linear(16, 16),
        #     # nn.Tanh(),
        #     nn.Linear(16, 2)
        # )


    def forward(self, z):
        x = z[:, :d]
        # x = z[:, :d]
        u = z[:, d]

        dx = self.forward_x(z)
        dx += self.forward_u(z)
        return dx

    def forward_x(self, z):
        dx = torch.zeros_like(z[:, :2])
        # dx[:, 0]= z[:, 1]
        dx= self.a_net(self.transform(z))
        return dx

    def forward_u(self, z):
        u = z[:, d]
        B = torch.tensor(self.get_B(z), dtype=torch.float)
        # return self.B_net(z).view(d, m)@u
        return B@u

    def get_B(self, z):
        # return self.B_net(z).view(d, m).detach().numpy()
        return np.array([0, 1]).reshape(d, m)
