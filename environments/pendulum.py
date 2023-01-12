import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

from environments.environment import Environment


class Pendulum(Environment):

    d, m = 2, 1

    @staticmethod
    def observe(x):
        phi, d_phi = torch.unbind(x, dim=1)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        obs = torch.stack((cphi, sphi, d_phi), dim=1)
        return obs

    @staticmethod
    def get_state(obs):
        cphi, sphi, d_phi = torch.unbind(obs, dim=1)
        phi = torch.atan2(sphi, cphi)
        x = torch.stack((phi, d_phi), dim=1)
        return x

    def __init__(self, dt, sigma, gamma, mass, g, l, alpha):
        self.x0 = np.array([1e-3, 0.0])
        super().__init__(self.x0, self.d, self.m, dt, sigma, gamma)
        self.mass = mass
        self.g = g
        self.l = l
        self.inertia = (1/3)*mass*l**2
        self.omega_2 = mass*l*g/(2*self.inertia)
        self.alpha = alpha
        self.period = 2*np.pi / np.sqrt(self.omega_2)


        self.n_points = 10

        self.phi_max = np.pi
        self.dphi_max = np.sqrt(-2*self.omega_2*np.cos(self.phi_max))
        self.dphi_max = 8.0

        self.x_lim = np.array([[-np.inf, np.inf], [-self.dphi_max, self.dphi_max]])
        self.x_min, self.x_max = self.x_lim[:, 0], self.x_lim[:, 1]

        self.B_star = np.array([[0.0], [(1/self.inertia)]])
        self.A_star = np.array([
            [0, 0, 1],
            [0, -self.omega_2, -self.alpha]
        ])
        self.theta_star = np.hstack((self.A_star, self.B_star))

        self.goal_weights = torch.Tensor((10., 10., 50.))
        self.goal_weights_relaxed = torch.Tensor((1., 1., 0.1))
        # self.goal_weights_relaxed = torch.Tensor((1., 1., 0.1))
        # self.goal_weights = torch.Tensor((100, .1, 0.1))
        self.goal_state = torch.Tensor((-1., 0., 0.))
        self.R = 0.001


    def dynamics(self, x, u):
        phi, d_phi = x
        cphi, sphi = np.cos(phi), np.sin(phi)
        obs = np.array([cphi, sphi, d_phi])

        x_dot = self.A_star @ obs + self.B_star @ u
        return x_dot
    

    def d_dynamics(self, z):
        # x_dot[0] = torch.clip(x[1], -8, 8)
        phi, d_phi, u = torch.unbind(z, dim=1)
        # print(f'u = {u.shape}')
        # print(f'phi = {phi.shape}')
        sphi = torch.sin(phi)
        dd_phi = -self.omega_2*sphi -self.alpha*d_phi + (1/self.inertia)*u.squeeze()
        x_dot = torch.stack((d_phi, dd_phi), dim=1)
        return x_dot
    

    def plot_system(self, x, u, t):
        phi = x[0]
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        plt.plot([0, self.l*s_phi], [0, -self.l*c_phi], color='blue')
        plt.xlim((-2*self.l, 2*self.l))
        plt.ylim((-2*self.l, 2*self.l))
        plt.gca().set_aspect('equal', adjustable='box')
        # if u > 0:
        #     theta1, theta2 = -50, 180
        # else:
        #     theta1, theta2 = 180, -50

        # draw_self_loop(plt.gca(), (0.0, 0.0), 0.1*self.l, 'red', 'white', theta1, theta2)
        # circle = plt.Circle((0, 0), 0.03, color='black')
        # plt.gca().add_patch(circle)
        plt.arrow(
            self.l*s_phi,
            -self.l*c_phi,
            0.1*u[0]*c_phi,
            0.1*u[0]*s_phi,
            color='red',
            head_width=0.1,
            head_length=0.1*abs(u[0]),
            alpha=0.5)
        plt.xticks([]) ; plt.yticks([])
        plt.title(rf'$t = {t}$')

        # plt.title('periodic inputs')
        # plt.savefig(f'output/animations/pendulum/periodic-{t//5}.png')

    def plot_phase(self, x):
        plt.scatter(x[0], x[1])
        plt.xlim((-self.phi_max, self.phi_max))
        plt.ylim((-self.dphi_max, self.dphi_max))

    def plot_portrait(self, f, grid_phi, grid_dphi):
        grid = torch.cat([
            grid_phi.reshape(-1, 1),
            grid_dphi.reshape(-1, 1),
        ], 1)
        predictions = f(grid)

        n_points, _ = grid_phi.shape

        # plt.xlim((-self.phi_max, self.phi_max))
        # plt.ylim((-self.dphi_max, self.dphi_max))
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
            linewidth=linewidth*2,
            arrowsize=.8,
            density=.6)


class DampedPendulum(Pendulum):

    def __init__(self, dt=5e-2, sigma=1e-3):
        # sigma = .1
        gamma = 1.0
        mass = 1.0
        l = 1.0
        g = 10.0
        alpha = 1.
        super().__init__(dt, sigma, gamma, mass, g, l, alpha)



class GymPendulum(Pendulum):

    def __init__(self, dt=80e-3):
        sigma = 0.
        gamma = 2.0
        mass = 1.0
        l = 1.0
        g = 10.0
        alpha = 0.0
        super().__init__(dt, sigma, gamma, mass, g, l, alpha)

class DmPendulum(Pendulum):

    def __init__(self, dt=80e-3, sigma=0):
        mass = 1.0
        l = 0.5
        g = 10.0
        alpha = 0.1
        gamma = (1/6)*mass*g*l/2
        super().__init__(dt, sigma, gamma, mass, g, l, alpha)


class LinearizedPendulum(Pendulum):

    def __init__(self, dt=2e-2):
        sigma = 0.2
        gamma = 0.5
        mass = 1.0
        l = 1.0
        g = 10.0
        alpha = .5
        super().__init__(dt, sigma, gamma, mass, g, l, alpha)

        # self.period = 2*self.period

        self.dphi_max = np.sqrt(self.omega_2*2)

        self.A = np.array([
            [0, 1],
            [-self.omega_2, -alpha]
        ])
        self.B = np.array([[0], [1/self.inertia]])
        self.n_points = 10
        self.phi_max = np.pi
    
    def plot_system(self, x, u, t):
        super().plot_system(x, 0.5*u, t)
        plt.xlim((-.8*self.l, .8*self.l))
        plt.ylim((-1.2*self.l, .1*self.l))

    def dynamics(self, x, u):
        x_dot = self.A @ x + self.B@u
        # x_dot += np.array([[0.0], [1/self.inertia]]) @ u
        # x_dot[1] = d_phi
        # x_dot[1] = np.clip(d_phi, -10, 10)
        
        return x_dot



    # plt.gca().set_aspect('equal', adjustable='box')
