import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class XGrid:

    def __init__(self, environment, n_points=10):

        self.environment = environment
        self.plot_system = environment.plot_system

        self.n_points = n_points

        self.phi_max = environment.phi_max
        self.dphi_max = environment.dphi_max

        self.interval_phi = torch.linspace(-self.phi_max, self.phi_max, self.n_points)
        self.interval_dphi = torch.linspace(-self.dphi_max,
                                            self.dphi_max, self.n_points)
        # self.interval_u = torch.linspace(-self.environment.gamma,
                                        #  self.environment.gamma, 5)
        # self.interval_u = torch.linspace(-environment.gamma, environment.gamma, 2)
        
        self.loss_function = nn.MSELoss(reduction='mean')

       
        grid_phi1, grid_phi2, grid_d_phi1, grid_d_phi2 = torch.meshgrid(
            self.interval_phi,
            self.interval_phi,
            self.interval_dphi,
            self.interval_dphi,
        )
        self.grid = torch.cat([
            torch.cos(grid_phi1.reshape(-1, 1)),
            torch.sin(grid_phi1.reshape(-1, 1)),
            torch.cos(grid_phi2.reshape(-1, 1)),
            torch.sin(grid_phi2.reshape(-1, 1)),
            grid_d_phi1.reshape(-1, 1),
            grid_d_phi2.reshape(-1, 1)
        ], 1)

        # self.grid = 10*torch.randn(10, 2)


    def evaluate(self, model, t):
        # print("evaluate")
        predictions = model.predict(self.grid.clone())
        truth = self.f_star(self.grid.clone())  
        # print(f'predictions {predictions}')
        # print(f'truth {truth}')
        # print(f'error {(truth-predictions)[120:150]}')

        # print(predictions)

        loss = self.loss_function(predictions, truth)
        return loss


    def f_star(self, obs):
        cphi1, sphi1, cphi2, sphi2, d_phi1, d_phi2 = torch.unbind(obs, dim=1)
        cdelta = cphi1*cphi2 + sphi1*sphi2
        sdelta = sphi1*cphi2 - sphi2*cphi1
        s2delta = 2*sdelta*cdelta
        dd_phi1, dd_phi2 = self.environment.acceleration_x(cphi1, cphi2, sphi1, cdelta, sdelta, s2delta, d_phi1, d_phi2)
        dd = torch.stack((dd_phi1, dd_phi2), dim=1)
        # dx[:, 0] = z[:, 1]
        # dx[:, 2] = z[:, 1]
        # dd[:, 0] = dd_y
        # dd[:, 1] = dd_phi
        return dd


# class ZGrid(GridEvaluation):
#     def __init__(self, environment):
#         super().__init__(environment)
#         grid_dy, grid_phi, grid_dphi, grid_u = torch.meshgrid(
#             self.interval_dy,
#             self.interval_phi,
#             self.interval_dphi,
#             self.interval_u
#         )
#         self.grid = torch.cat([
#             grid_dy.reshape(-1, 1),
#             torch.cos(grid_phi.reshape(-1, 1)),
#             torch.sin(grid_phi.reshape(-1, 1)),
#             grid_dphi.reshape(-1, 1),
#             grid_u.reshape(-1, 1),
#         ], 1)

#     def f_star(self, zeta_u):
#         d_y, c_phi, s_phi, d_phi, u = torch.unbind(zeta_u, dim=1)
#         dd_y, dd_phi = self.environment.acc(
#             d_y, c_phi, s_phi, d_phi, u)
#         # dx[:, 0] = z[:, 1]
#         # dx[:, 2] = z[:, 1]
#         # dd = torch.stack((d_y, dd_y, d_phi, dd_phi), dim=1)
#         dd = torch.stack((dd_y, dd_phi), dim=1)
#         return dd


# def f_star(zeta):
#     c_phi1, c_phi2, s_phi1, s_phi2 = zeta[:,
#                                           0], zeta[:, 1], zeta[:, 2], zeta[:, 3]
#     d_phi1, d_phi2 = zeta[:, 4], zeta[:, 5]
#     c_dphi = c_phi1*c_phi2 + s_phi1*s_phi2
#     s_dphi = s_phi1*c_phi2 - c_phi1*s_phi2
#     s_2dphi = 2*s_dphi*c_dphi
#     dx = torch.zeros_like(zeta[:, :2])
#     a_phi1, a_phi2 = acceleration_x(
#         c_phi1, c_phi2, s_phi1, s_phi2, c_dphi, s_dphi, s_2dphi, d_phi1, d_phi2)
#     dx[:, 0] = a_phi1
#     dx[:, 1] = a_phi2
#     return dx





# def test_error(model, x, u, plot, t=0):
#     truth = f_star(grid)
#     loss_function = nn.MSELoss()
#     predictions = model.net(grid.clone()).squeeze()
#     # # # print(f'prediction {predictions.shape} target {truth.shape} ')
#     loss = loss_function(predictions, truth)
#     if plot and t % 10 == 0:
#         plt.title(t)
#         # plot_portrait(model.net)
#         plt.pause(0.1)
#         plt.close()
#     # print(f'loss = {loss}')
#     return loss
