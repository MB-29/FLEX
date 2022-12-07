import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class Grid:

    def __init__(self, environment, n_points=20):

        self.environment = environment
        self.plot_system = environment.plot_system
        
        self.n_points = n_points

        self.dy_max = environment.dy_max
        self.dphi_max = environment.dphi_max

        self.interval_dy = torch.linspace(-self.dy_max, self.dy_max, self.n_points)
        self.interval_phi = torch.linspace(-np.pi, np.pi, self.n_points)
        self.interval_dphi = torch.linspace(-self.dphi_max,
                                       self.dphi_max, self.n_points)
        self.interval_u = torch.linspace(-self.environment.gamma, self.environment.gamma, 5)
        # self.interval_u = torch.linspace(-environment.gamma, environment.gamma, 2)
        
  

    def test_error(self, model, x, u, plot, t=0):
        truth = self.f_star(self.grid)
        loss_function = nn.MSELoss()
        predictions = model.predict(self.grid.clone()).squeeze()
        # # print(f'prediction {predictions.shape} target {truth.shape} ')
        loss = loss_function(predictions, truth)
        # loss = torch.linalg.norm(self.A_star-model.a_net[0].weight)
        if plot and t%2 ==0:
            self.plot_system(x, u, t)
            # plot_portrait(model.forward_x)
            plt.pause(0.1)
            plt.close()
        # print(f'loss = {loss}')
        # print(x)
        return loss

        
class XGrid(Grid):
    def __init__(self, environment):
        super().__init__(environment)
        grid_dy, grid_phi, grid_dphi = torch.meshgrid(
            self.interval_dy,
            self.interval_phi,
            self.interval_dphi
        )
        self.grid = torch.cat([
            grid_dy.reshape(-1, 1),
            torch.cos(grid_phi.reshape(-1, 1)),
            torch.sin(grid_phi.reshape(-1, 1)), 
            grid_dphi.reshape(-1, 1)
            # grid_u.reshape(-1, 1),
        ], 1)
    def f_star(self, zeta):
        d_y, c_phi, s_phi, d_phi = torch.unbind(zeta, dim=1)
        u = torch.zeros_like(d_y)
        dd_y, dd_phi = self.environment.acc(
            d_y, c_phi, s_phi, d_phi, u)
        dd = torch.stack((dd_y, dd_phi), dim=1)
        # dx[:, 0] = z[:, 1]
        # dx[:, 2] = z[:, 1]
        # dd[:, 0] = dd_y
        # dd[:, 1] = dd_phi
        return dd

        
class ZGrid(Grid):
    def __init__(self, environment):
        super().__init__(environment)
        grid_dy, grid_phi, grid_dphi, grid_u = torch.meshgrid(
                self.interval_dy,
                self.interval_phi,
                self.interval_dphi,
                self.interval_u
            )
        self.grid = torch.cat([
            grid_dy.reshape(-1, 1),
            torch.cos(grid_phi.reshape(-1, 1)),
            torch.sin(grid_phi.reshape(-1, 1)),
            grid_dphi.reshape(-1, 1),
            grid_u.reshape(-1, 1),
        ], 1)
    def f_star(self, zeta_u):
        d_y, c_phi, s_phi, d_phi, u = torch.unbind(zeta_u, dim=1)
        dd_y, dd_phi = self.environment.acc(
            d_y, c_phi, s_phi, d_phi, u)
        # dx[:, 0] = z[:, 1]
        # dx[:, 2] = z[:, 1]
        # dd = torch.stack((d_y, dd_y, d_phi, dd_phi), dim=1)
        dd = torch.stack((dd_y, dd_phi), dim=1)
        return dd


