import numpy as np
import torch
import torch.nn as nn

from models.models import Model
from evaluation.pendulum import NormA, ParameterNorm


class Model(Model):

    def __init__(self, environment, evaluation=None):
        super().__init__(environment, evaluation)
        self.t_period = environment.period / environment.dt
        self.period = environment.period
        self.B_star = torch.tensor(environment.B_star, dtype=torch.float)
        self.d, self.m = environment.d, environment.m
        self.evaluation = evaluation

        self.linear = False

    def forward_x(self, x):
        phi, d_phi = torch.unbind(x, dim=1)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        obs = torch.stack((cphi, sphi, d_phi), dim=1)
        return self.net(obs)

    def forward(self, z):
        x = z[:, :self.d]
        u = z[:, self.d:]
        phi, d_phi, u = torch.unbind(z, dim=1)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        obs = torch.stack((cphi, sphi, d_phi), dim=1)
        obs_u = torch.cat((obs, u.unsqueeze(1)), dim=1)
        x_dot = self.predict(obs_u)
        return x_dot


class Linear(Model):

    def __init__(self, environment, evaluation=None):

        evaluation = ParameterNorm(
            environment) if evaluation is None else evaluation
        super().__init__(environment, evaluation)
        self.net = nn.Sequential(
            nn.Linear(4, 2, bias=False),
        )

    def predict(self, obs_u):
        x_dot = self.net(obs_u)
        return x_dot

class LinearReduced(Model):

    def __init__(self, environment, evaluation=None):
        evaluation = NormA(environment) if evaluation is None else evaluation
        super().__init__(environment, evaluation)
        self.net = nn.Sequential(
            nn.Linear(3, 2, bias=False),
        )

    def predict(self, obs_u):
        obs = obs_u[:, :-1]
        u = obs_u[:, -1:]
        x_dot = self.net(obs) + (self.B_star @ u.T).T
        return x_dot
