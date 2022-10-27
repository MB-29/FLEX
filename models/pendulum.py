import numpy as np
import torch
import torch.nn as nn

import environments.pendulum as pendulum

d, m = pendulum.d, pendulum.m


class Model(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.period = environment.period

    def get_B(self, x):
        B = np.zeros((d, m))
        B[1, 0] = 1
        return B

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

    def __init__(self, environment):
        super().__init__(environment)
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            # nn.Linear(16, 16),
            # nn.Tanh(),
            nn.Linear(16, 1)
        )
        self.lr = 0.005

    def forward_x(self, x):
        dx = torch.zeros_like(x)
        dx[:, 0] = x[:, 1]
        zeta = x.clone()
        zeta[:, 0] = torch.sin(x[:, 0])
        # x[:, 1] = torch.sin(x[:, 1])
        dx[:, 1] = self.net(zeta).view(-1)
        return dx

# class LinearModel(Model):
#     def __init__(self, environment):
#         super().__init__(environment)
#         self.theta = nn.parameter.Parameter(torch.zeros(2, dtype=torch.float))
#         self.lr = 0.01
    
#     def forward_x(self, x):
#         dx = torch.zeros_like(x)
#         dx[:, 0] = x[:, 1]
#         dx[:, 1] = self.theta[0] * torch.sin(x[:, 0]) + self.theta[1]*x[:, 1]
#         return dx


class GymNeural(Model):

    def __init__(self, environment):
        super().__init__(environment)

        self.net = nn.Sequential(
            nn.Linear(4, 5),
            nn.Tanh(),
            # nn.Linear(5, 5),
            # nn.Tanh(),
            nn.Linear(5, 2)
        )

    def transform(self, x):
        batch_size, _ = x.shape
        zeta = torch.zeros(batch_size, 3)
        zeta[:, 0] = torch.cos(x[:, 0])
        zeta[:, 1] = torch.sin(x[:, 0])
        zeta[:, 2] = x[:, 1]
        return zeta

    def forward(self, z):
        batch_size, _ = z.shape
        x = z[:, :d]
        u = z[:, d:]
        zeta = self.transform(x)

        zeta_u = torch.zeros((batch_size, d+1+m))
        zeta_u[:, :d+1] = zeta
        zeta_u[:, d+1:] = u
        dx = self.net(zeta_u)

        return dx


class LinearModel(NeuralModel):

    def __init__(self, environment):
        super().__init__(environment)
        self.net = nn.Sequential(
            nn.Linear(2, 2, bias=False),
        )
        self.lr = 0.01
        self.B_star = torch.tensor(environment.B_star)

    def forward(self, z):
        x = z[:, :d]
        u = z[:, d:]
        dx = self.net(x) + (self.B_star @ u.T).T
        return dx