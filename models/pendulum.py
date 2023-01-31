import numpy as np
import torch
import torch.nn as nn

from evaluation.pendulum import GridAB, NormA, NormTheta


class Model(nn.Module):

    def __init__(self, environment, evaluation=None):
        super().__init__()
        self.t_period = environment.period / environment.dt
        self.period = environment.period
        self.B_star = torch.tensor(environment.B_star, dtype=torch.float)
        self.d, self.m = environment.d, environment.m
        self.evaluation = evaluation

        self.linear = False

    def transform(self, z):
        return z[:, :self.d]

    def forward_x(self, x):
        phi, d_phi = torch.unbind(x, dim=1)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        zeta = torch.stack((cphi, sphi, d_phi), dim=1)
        return self.net(zeta)

class NeuralModel(Model):

    def __init__(self, environment):
        super().__init__(environment)
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        self.lr = 0.005

    def predict(self, x):
        return self.forward_x(x)

    def forward_u(self, dx, u):
        dx += (self.B_star @ u.unsqueeze(1).T).T
        return dx

    def forward(self, z):
        x = self.transform(z)
        u = z[:, self.d]

        x_dot = self.forward_x(x)
        x_dot = self.forward_u(x_dot, u)
        return x_dot

    def forward_x(self, x):
        x_dot = torch.zeros_like(x)
        x_dot[:, 0] = x[:, 1]
        x_dot[:, 1] = self.net(x).view(-1)
        return x_dot


class NeuralAB(Model):


    def __init__(self, environment, evaluation = None):
        evaluation = GridAB(environment) if evaluation is None else evaluation
        super().__init__(environment, evaluation)

        self.net = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
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
        x = z[:, :self.d]
        u = z[:, self.d:]
        phi, d_phi, u = torch.unbind(z, dim=1)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        zeta = torch.stack((cphi, sphi, d_phi), dim=1)
        zeta_u = torch.cat((zeta, u.unsqueeze(1)), dim=1)
        x_dot = self.predict(zeta_u)
        return x_dot

    def predict(self, zeta_u):
        cphi, sphi, d_phi, u = torch.unbind(zeta_u, dim=1)
        x_dot = self.net(zeta_u)
        return x_dot 

class LinearA(NeuralAB):

    def __init__(self, environment, evaluation=None):
        evaluation = NormA(environment) if evaluation is None else evaluation
        super().__init__(environment, evaluation)
        self.net = nn.Sequential(
            nn.Linear(3, 2, bias=False),
        )

    def predict(self, zeta_u):
        zeta = zeta_u[:, :-1]
        u = zeta_u[:, -1:]
        x_dot = self.net(zeta) + (self.B_star @ u.T).T
        return x_dot

class LinearAB(NeuralAB):

    def __init__(self, environment, evaluation=None):
        
        evaluation = NormTheta(environment) if evaluation is None else evaluation
        super().__init__(environment, evaluation)
        self.net = nn.Sequential(
            nn.Linear(4, 2, bias=False),
        )

    def predict(self, zeta_u):
        zeta = zeta_u[:, :-1]
        u = zeta_u[:, -1:]
        x_dot = self.net(zeta_u)
        return x_dot