import numpy as np
import torch
import torch.nn as nn

import environments.pendulum_gym as pendulum_gym

d, m = pendulum_gym.d, pendulum_gym.m
dt = pendulum_gym.dt


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def transform(self, x):
        batch_size, _ = x.shape
        zeta = torch.zeros(batch_size, 3)
        zeta[:, 0] = torch.cos(x[:, 0])
        zeta[:, 1] = torch.sin(x[:, 0])
        zeta[:, 2] = x[:, 1]
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
    
        self.B_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            # nn.Linear(16, 16),
            # nn.Tanh(),
            nn.Linear(16, 2)
        )


    def forward(self, z):
        x = z[:, :d]
        # x = z[:, :d]
        u = z[:, d]

        dx = self.forward_x(z)
        dx += self.forward_u(z)
        return dx
    
    def model(self, x, u):
        z = torch.zeros(1, d+m)
        z[:, :d] = x
        z[:, d] = u
        return self.forward(z).squeeze()

    def forward_x(self, z):
        x = z[:, :d]
        dx = torch.zeros_like(z[:, :2])
        # dx[:, 0]= z[:, 1]
        dx= self.a_net(self.transform(x))
        return dx

    def forward_u(self, z):
        x = z[:, :d]
        u = z[:, d]
        B = torch.tensor(self.get_B(x.detach().numpy().squeeze()), dtype=torch.float)
        return self.B_net(z).view(d, m)@u
        return B@u

    def get_B(self, x):
        zeta = self.transform(torch.tensor(x, dtype=torch.float).unsqueeze(0))
        return self.B_net(zeta).view(d, m).detach().numpy()
        return np.array([0, 1]).reshape(d, m)
