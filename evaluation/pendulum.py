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
        # predictions = model.predict(self.grid.clone())
        # truth = self.f_star(self.grid.clone())
        # # print(predictions)

        # loss = self.loss_function(predictions, truth)
        return 0


class XGrid(GridEvaluation):
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

        # x_dot = torch.stack((d_phi, dd_phi), dim=1)                               
        return x_dot

# class Animations(XGrid):
#     def __init__(self, environment):
#         super().__init__(environment)
#     def evaluate(self, model, t):
#         rate = 3

#         loss = torch.zeros(1)
#         print(f't={t}')
#         fig = plt.figure(1, figsize=(12, 4))
#         fig.set_tight_layout(True)
#         fig.suptitle(fr'exploration, $t={t}$')
#         # fig.suptitle(fr'{agent_name} inputs, $t={t}$')

#         plt.subplot(131)
#         plt.title('state')
#         self.environment.plot_system(x, u, t)
#         # plt.savefig(f'output/animations/pendulum/{agent_name}-{t//rate}.pdf')
#         # if plot:
#         #     plt.pause(0.1)
#         # plt.close()

#         plt.subplot(132)
#         plt.title('trajectory')
#         self.environment.plot_portrait(
#             self.environment.f_star, self..grid_phi, self..grid_dphi)
#         plt.plot(agent.x_values[:t:3, 0], agent.x_values[:t:3,
#                                                          1], alpha=0.7, lw=2, color=color)
#         plt.xlabel(r'$\varphi$')
#         plt.ylabel(r'$\dot{\varphi}$', rotation=0)
#         plt.gca().yaxis.set_label_coords(-0.15, .5)
#         plt.xticks([-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$\pi$'])
#         plt.yticks([-self.environment.dphi_max, 0, self.environment.dphi_max],
#                    [r'$-\omega_0$', r'$0$', r'$\omega_0$'])

#         plt.subplot(133)
#         # plt.title(r'$f_\theta$')
#         plt.title(r'learned model')
#         # self.environment.plot_error(model.forward_x)
#         self.environment.plot_portrait(
#             model.forward_x, self..grid_phi, self..grid_dphi)
#         # plt.scatter(x[0], x[1], color=color)
#         plt.xlabel(r'$\varphi$')
#         plt.ylabel(r'$\dot{\varphi}$', rotation=0)
#         plt.gca().yaxis.set_label_coords(-0.15, .5)
#         plt.xticks([-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$\pi$'])
#         plt.yticks([-self.environment.dphi_max, 0, self.environment.dphi_max],
#                    [r'$-\omega_0$', r'$0$', r'$\omega_0$'])

#         plt.savefig(f'output/animations/pendulum/periodic-{t//rate}.pdf')
#         plt.pause(0.1)
#         plt.close()



class ZGrid(GridEvaluation):
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


class NormTheta(Evaluation):

    def __init__(self, environment):
        super().__init__(environment)
        self.theta_star = torch.tensor(environment.theta_star, dtype=torch.float)

    def evaluate(self, model, t):

        loss = torch.linalg.norm(self.theta_star-model.net[0].weight)
        return loss
