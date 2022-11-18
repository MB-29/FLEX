import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class Evaluation:

    def __init__(self, environment):

        self.environment = environment
        

        self.n_points = 10

        self.phi_max = np.pi
        self.dphi_max = np.sqrt(-2*environment.omega_2*np.cos(environment.phi_max))
        self.dphi_max = 8.0
        self.interval_phi = torch.linspace(-self.phi_max,
                                        self.phi_max, self.n_points)
        self.interval_dphi = torch.linspace(-self.dphi_max,
                                            self.dphi_max, self.n_points)
        interval_u = torch.linspace(-environment.gamma, environment.gamma, self.n_points)
        self.grid_phi, self.grid_dphi, grid_u = torch.meshgrid(
            self.interval_phi, self.interval_dphi, interval_u)
        self.grid = torch.cat([
            torch.cos(self.grid_phi.reshape(-1, 1)),
            torch.sin(self.grid_phi.reshape(-1, 1)),
            self.grid_dphi.reshape(-1, 1),
            grid_u.reshape(-1, 1)
        ], 1)


        self.plot_system = environment.plot_system

    def f_star(self, zeta_u):
        dx = torch.zeros_like(zeta_u[:, :self.environment.d])
        u = zeta_u[:, -1]
        # print(zeta_u[:, 1].shape)
        dx[:, 0] = zeta_u[:, 2]
        dx[:, 1] = (1/self.environment.inertia)*(-(1/2)*self.environment.m *
                                     self.environment.g*self.environment.l*zeta_u[:, 1] + u)
        return dx

    def test_error(self, model, x, u, plot, t=0):
        loss_function = nn.MSELoss()

        # predictions = model.a_net(self.grid.clone()).squeeze()
        # truth = self.f_star(self.grid)
        # print(f'prediction {predictions.shape} target {truth.shape} ')

        # predictions = model.B_net(self.grid.clone())
        # batch_size, _ = predictions.shape
        # truth = self.B_star.view(1, 2).expand(batch_size,  -1)

        predictions = model.predict(self.grid.clone())
        batch_size, _ = predictions.shape
        truth = self.f_star(self.grid.clone())
        # print(predictions)

        loss = loss_function(predictions, truth)
        # loss = torch.linalg.norm(self.A_star-model.net[0].weight)
        if plot and t % 2 == 0:
            self.plot_system(x, u, t)
            # plot_portrait(model.forward_x)
            plt.pause(0.1)
            plt.close()
        # print(f'loss = {loss}')
        # print(x)
        return loss
