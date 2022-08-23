import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

from environments.environment import Environment

d, m = 2, 1


class Pendulum(Environment):

    def __init__(self, d, m, dt, sigma, gamma, mass, g, l, alpha):
        super().__init__(d, m, dt, sigma, gamma)
        self.mass = mass
        self.g = g
        self.l = l
        self.omega_2 = g/l
        self.inertia = (1/3)*m*l**2
        self.alpha = alpha
        self.period = 2*np.pi / np.sqrt(self.omega_2)
        self.A_star = torch.tensor([
            [0, 0, 1],
            [0, -(3/2)*self.omega_2, 0]
        ])

        self.x0 = np.array([0.0, 0.0])

        n_points = 100


        self.phi_max = np.pi
        self.dphi_max = 2*np.sqrt(-2*self.omega_2*np.cos(self.phi_max))
        interval_phi = torch.linspace(-self.phi_max, self.phi_max, n_points)
        interval_dphi = torch.linspace(-self.dphi_max, self.dphi_max, n_points)
        grid_phi, grid_dphi = torch.meshgrid(interval_phi, interval_dphi)
        self.grid = torch.cat([
            torch.cos(grid_phi.reshape(-1, 1)),
            torch.sin(grid_phi.reshape(-1, 1)),
            grid_dphi.reshape(-1, 1)
        ], 1)

    def dynamics(self, x, u):
        dx = np.zeros_like(x)
        dx[0] = x[1]
        d_phi = (1/self.inertia)*(-(1/2)*self.m*self.g*self.l * np.sin(x[0]) + u)
        dx[1] = d_phi
        # dx[1] = np.clip(d_phi, -10, 10)
        noise = self.sigma * np.random.randn(d)
        dx += noise
        return dx


    def d_dynamics(self, x, u):
        dx = torch.zeros_like(x)
        # dx[0] = torch.clip(x[1], -8, 8)
        dx[0] = x[1]
        dx[1] = (1/self.inertia) * (-(1/2)*self.m*self.g*self.l * torch.sin(x[0]) + u)
        return dx

    def f_star(self, zeta):
        dx = torch.zeros_like(zeta[:, :d])
        dx[:, 0] = zeta[:, 2]
        dx[:, 1] = -(1/self.inertia)*(1/2)*self.m*self.g*self.l*zeta[:, 1]
        return dx


    def step_cost(self, x, u):
        c_phi, s_phi = torch.cos(x[0]), torch.sin(x[0])
        d_phi = x[1]
        c = 100*(c_phi + 1)**2 + 0.1*s_phi**2 + 0.1*d_phi**2 + 0.001*u**2
        # print(f'x = {x}, u={u}, c={c}')
        return c

    def test_error(self, model, x, u, plot, t=0):
        truth = self.f_star(self.grid)
        loss_function = nn.MSELoss()
        predictions = model.a_net(self.grid.clone()).squeeze()
        # # print(f'prediction {predictions.shape} target {truth.shape} ')
        loss = loss_function(predictions, truth)
        # loss = torch.linalg.norm(self.A_star-model.a_net[0].weight)
        if plot and t%5 == 0:
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
        d, m = 2, 1
        sigma = 0.1
        gamma = 0.5
        mass = 1.0
        l = 1.0
        g = 10.0
        alpha = 1.0
        super().__init__(d, m, dt, sigma, gamma, mass, g, l, alpha)

class GymPendulum(Pendulum):

    def __init__(self, dt):
        d, m = 2, 1
        sigma = 0
        gamma = 2
        mass = 1.0
        l = 1.0
        g = 10.0
        alpha = 0.0
        super().__init__(d, m, dt, sigma, gamma, mass, g, l, alpha)

def plot_phase(x):
    plt.scatter(x[0], x[1])
    plt.xlim((-phi_max, phi_max))
    plt.ylim((-dphi_max, dphi_max))
    plt.gca().set_aspect('equal', adjustable='box')



def plot_portrait(f):
    predictions = f(grid)

    plt.xlim((-phi_max, phi_max))
    plt.ylim((-dphi_max, dphi_max))
    vector_x = predictions[:, 0].reshape(n_points, n_points).detach().numpy()
    vector_y = predictions[:, 1].reshape(n_points, n_points).detach().numpy()
    magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
    linewidth = magnitude / magnitude.max()
    plt.streamplot(
        grid_phi.numpy().T,
        grid_dphi.numpy().T,
        vector_x.T,
        vector_y.T,
        color='black',
        linewidth=linewidth)
    # plt.gca().set_aspect('equal', adjustable='box')
