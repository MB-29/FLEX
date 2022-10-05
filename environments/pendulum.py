import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

from environments.environment import Environment

d, m = 2, 1


class Pendulum(Environment):

    def __init__(self, dt, sigma, gamma, mass, g, l, alpha):
        super().__init__(d, m, dt, sigma, gamma)
        self.mass = mass
        self.g = g
        self.l = l
        self.inertia = (1/3)*m*l**2
        self.omega_2 = g/(2*l*self.inertia)
        self.alpha = alpha
        self.period = 2*np.pi / np.sqrt(self.omega_2)

        self.x0 = np.array([0.0, 0.0])

        self.n_points = 10

        self.phi_max = np.pi
        self.dphi_max = 2*np.sqrt(-2*self.omega_2*np.cos(self.phi_max))
        self.interval_phi = torch.linspace(-self.phi_max,
                                      self.phi_max, self.n_points)
        self.interval_dphi = torch.linspace(-self.dphi_max,
                                       self.dphi_max, self.n_points)
        interval_u = torch.linspace(-self.gamma, self.gamma, self.n_points)
        self.grid_phi, self.grid_dphi, grid_u = torch.meshgrid(
            self.interval_phi, self.interval_dphi, interval_u)
        self.grid = torch.cat([
            torch.cos(self.grid_phi.reshape(-1, 1)),
            torch.sin(self.grid_phi.reshape(-1, 1)),
            self.grid_dphi.reshape(-1, 1),
            grid_u.reshape(-1, 1)
        ], 1)

        self.B_star = torch.tensor([[0.0], [(1/self.inertia)]])
        self.A_star = torch.tensor([
            [0, 0, 1],
            [0, -(3/2)*self.omega_2, 0]
        ])

    def dynamics(self, x, u):
        dx = np.zeros_like(x)
        dx[0] = x[1]
        dx[1] = (1/self.inertia)*(-(1/2)*self.m *
                                  self.g*self.l * np.sin(x[0]) + u)
        # dx += np.array([[0.0], [1/self.inertia]]) @ u
        # dx[1] = d_phi
        # dx[1] = np.clip(d_phi, -10, 10)
        noise = self.sigma * np.random.randn(d)
        dx += noise
        return dx

    def a(self, x):
        dx = np.zeros_like(x)
        dx[0] = x[1]
        dx[1] = (1/self.inertia)*(-(1/2)*self.m*self.g*self.l * np.sin(x[0]))
        return dx

    def b(self, x, u):
        return np.array([[0.0], [1/self.inertia]]) @ u
    # def dynamics(self, x, u):
    #     dx = self.a(x) + self.b(x, u)
    #     return dx

    def d_dynamics(self, x, u):
        dx = torch.zeros_like(x)
        # dx[0] = torch.clip(x[1], -8, 8)
        dx[0] = x[1]
        dx[1] = (1/self.inertia) * (-(1/2)*self.m *
                                    self.g*self.l * torch.sin(x[0]) + u)
        return dx

    def f_star(self, zeta_u):
        dx = torch.zeros_like(zeta_u[:, :d])
        u = zeta_u[:, -1]
        # print(zeta_u[:, 1].shape)
        dx[:, 0] = zeta_u[:, 2]
        dx[:, 1] = (1/self.inertia)*(-(1/2)*self.m *
                                     self.g*self.l*zeta_u[:, 1] + u)
        return dx

    def step_cost(self, x, u):
        c_phi, s_phi = torch.cos(x[0]), torch.sin(x[0])
        d_phi = x[1]
        c = 100*(c_phi + 1)**2 + 0.1*s_phi**2 + 0.1*d_phi**2 + 0.001*u**2
        # print(f'x = {x}, u={u}, c={c}')
        return c

    def test_error(self, model, x, u, plot, t=0):
        loss_function = nn.MSELoss()

        # predictions = model.a_net(self.grid.clone()).squeeze()
        # truth = self.f_star(self.grid)
        # print(f'prediction {predictions.shape} target {truth.shape} ')

        # predictions = model.B_net(self.grid.clone())
        # batch_size, _ = predictions.shape
        # truth = self.B_star.view(1, 2).expand(batch_size,  -1)

        predictions = model.net(self.grid.clone())
        batch_size, _ = predictions.shape
        truth = self.f_star(self.grid.clone())
        # print(predictions)

        loss = loss_function(predictions, truth)
        # loss = torch.linalg.norm(self.A_star-model.a_net[0].weight)
        if plot and t % 5 == 0:
            self.plot_pendulum(x)
            # plot_portrait(model.forward_x)
            plt.pause(0.1)
            plt.close()
        # print(f'loss = {loss}')
        # print(x)
        return loss

    def plot_pendulum(self, x):
        phi = x[0]
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        plt.plot([0, self.l*s_phi], [0, -self.l*c_phi], color='blue')
        plt.xlim((-2*self.l, 2*self.l))
        plt.ylim((-2*self.l, 2*self.l))
        plt.gca().set_aspect('equal', adjustable='box')


class DampedPendulum(Pendulum):

    def __init__(self, dt):
        sigma = 0.1
        gamma = 0.5
        mass = 1.0
        l = 1.0
        g = 10.0
        alpha = 1.0
        super().__init__(dt, sigma, gamma, mass, g, l, alpha)

        self.grid = torch.cat([
            self.grid_phi.reshape(-1, 1),
            self.grid_dphi.reshape(-1, 1)
        ], 1)

    def f_star(self, x):
        dx = torch.zeros_like(x)
        dx[:, 0] = x[:, 1]
        dx[:, 1] = -self.omega_2 * torch.sin(x[:, 0]) - self.alpha*x[:, 1]
        return dx


class GymPendulum(Pendulum):

    def __init__(self, dt=80e-4):
        sigma = 0
        gamma = 2
        mass = 1.0
        l = 1.0
        g = 10.0
        alpha = 0.0
        super().__init__(dt, sigma, gamma, mass, g, l, alpha)


class LinearizedPendulum(Pendulum):

    def __init__(self, dt=0.01):
        sigma = 0.1
        gamma = 1.0
        mass = 1.0
        l = 1.0
        g = 10.0
        alpha = 1.0
        super().__init__(dt, sigma, gamma, mass, g, l, alpha)

        self.dphi_max = 2*np.sqrt(self.omega_2*2)

        self.A = np.array([
            [0, 1],
            [-self.omega_2, alpha]
        ])
        self.B = np.array([[0], [1]])

        self.grid_phi, self.grid_dphi = torch.meshgrid(
            self.interval_phi, self.interval_dphi)
        self.grid = torch.cat([
            self.grid_phi.reshape(-1, 1),
            self.grid_dphi.reshape(-1, 1)
        ], 1)

    def dynamics(self, x, u):
        dx = self.A @ x + self.B@u
        # dx += np.array([[0.0], [1/self.inertia]]) @ u
        # dx[1] = d_phi
        # dx[1] = np.clip(d_phi, -10, 10)
        noise = self.sigma * np.random.randn(d)
        dx += noise
        return dx

    def f_star(self, x):
        return (torch.tensor(self.A, dtype=torch.float)@x.T).T

    def test_error(self, model, x, u, plot, t=0):
        if not plot:
            return torch.tensor([0])
        print(x)
        self.plot_portrait(self.f_star)
        self.plot_phase(x)
        plt.pause(0.1)
        plt.close()

        return torch.tensor(0.0)

    def plot_phase(self, x):
        plt.scatter(x[0], x[1])
        plt.xlim((-self.phi_max, self.phi_max))
        plt.ylim((-self.dphi_max, self.dphi_max))

    def plot_portrait(self, f):
        predictions = f(self.grid)

        # plt.xlim((-self.phi_max, self.phi_max))
        # plt.ylim((-self.dphi_max, self.dphi_max))
        vector_x = predictions[:, 0].reshape(
            self.n_points, self.n_points).detach().numpy()
        vector_y = predictions[:, 1].reshape(
            self.n_points, self.n_points).detach().numpy()
        magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
        linewidth = magnitude / magnitude.max()
        plt.streamplot(
            self.grid_phi.numpy().T,
            self.grid_dphi.numpy().T,
            vector_x.T,
            vector_y.T,
            color='black',
            linewidth=linewidth)
    # plt.gca().set_aspect('equal', adjustable='box')
