import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class Evaluation:

    def __init__(self, environment):

        self.environment = environment
        self.plot_system = environment.plot_system

        self.n_points = 10

        self.phi_max = np.pi
        self.dphi_max = np.sqrt(-2*environment.omega_2 *
                                np.cos(environment.phi_max))
        self.dphi_max = 8.0
        self.interval_phi = torch.linspace(-self.phi_max,
                                           self.phi_max, self.n_points)
        self.interval_dphi = torch.linspace(-self.dphi_max,
                                            self.dphi_max, self.n_points)
        self.interval_u = torch.linspace(-environment.gamma,
                                         environment.gamma, self.n_points)


class GridEvaluation(Evaluation):

    def __init__(self, environment):
        super().__init__(environment)
        self.loss_function = nn.MSELoss()

    def evaluate(self, model, t):
        return 0


class GridA(GridEvaluation):
    def __init__(self, environment):
        super().__init__(environment)
        self.grid_phi, self.grid_dphi = torch.meshgrid(
            self.interval_phi, self.interval_dphi)
        self.grid = torch.cat([
            torch.cos(self.grid_phi.reshape(-1, 1)),
            torch.sin(self.grid_phi.reshape(-1, 1)),
            self.grid_dphi.reshape(-1, 1),
        ], 1)
        self.grid_x = torch.cat([
            self.grid_phi.reshape(-1, 1),
            self.grid_dphi.reshape(-1, 1),
        ], 1)

    def f_star(self, x):
        batch_size, _ = x.shape
        u = torch.zeros(batch_size, 1)
        z = torch.cat((x, u), dim=1)
        x_dot = self.environment.d_dynamics(z)

        return x_dot



class GridAB(GridEvaluation):
    def __init__(self, environment):
        super().__init__(environment)
        self.grid_phi, self.grid_dphi, grid_u = torch.meshgrid(
            self.interval_phi, self.interval_dphi, self.interval_u)
        self.grid = torch.cat([
            torch.cos(self.grid_phi.reshape(-1, 1)),
            torch.sin(self.grid_phi.reshape(-1, 1)),
            self.grid_dphi.reshape(-1, 1),
            grid_u.reshape(-1, 1)
        ], 1)
        self.grid_x = torch.cat([
            self.grid_phi.reshape(-1, 1),
            self.grid_dphi.reshape(-1, 1),
        ], 1)


    def f_star(self, zeta_u):
        dx = torch.zeros_like(zeta_u[:, :self.environment.d])
        u = zeta_u[:, -1]
        # print(zeta_u[:, 1].shape)
        dx[:, 0] = zeta_u[:, 2]
        dx[:, 1] = (1/self.environment.inertia)*(-(1/2)*self.environment.m *
                                                 self.environment.g*self.environment.l*zeta_u[:, 1] + u)
        return dx



class NormA(Evaluation):

    def __init__(self, environment):
        super().__init__(environment)
        self.A_star = torch.tensor(environment.A_star, dtype=torch.float)

    def evaluate(self, model, t):

        loss = torch.linalg.norm(self.A_star-model.net[0].weight[:, :3])
        return loss


class ParameterNorm(Evaluation):

    def __init__(self, environment):
        super().__init__(environment)
        self.theta_star = torch.tensor(environment.theta_star, dtype=torch.float)

    def evaluate(self, model, t):

        loss = torch.linalg.norm(self.theta_star-model.net[0].weight)
        return loss
