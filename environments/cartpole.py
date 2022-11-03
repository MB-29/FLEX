from re import S
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

from environments.environment import Environment


class Cartpole(Environment):

    d, m = 4, 1

    @staticmethod
    def observe(x):
        y, d_y, phi, d_phi = torch.unbind(x, dim=1)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        obs = torch.stack((y, d_y, cphi, sphi, d_phi), dim=1)
        return obs

    @staticmethod
    def get_state(obs):
        cphi, sphi, d_phi = torch.unbind(obs, dim=1)
        phi = torch.atan2(sphi, cphi)
        x = torch.stack((phi, d_phi), dim=1)

    def __init__(self, dt, sigma, gamma, g, mass, Mass, l, alpha, beta):
        super().__init__(self.d, self.m, dt, sigma, gamma)

        self.g = g
        self.l = l
        self.mass = mass
        self.Mass = Mass
        self.total_mass = Mass + mass
        self.alpha = alpha
        self.beta = beta

        self.omega_2 = g/l
        self.period = 2*np.pi * np.sqrt(l / g)
        self.x0 = np.array([0.0, 0.0, np.pi, 0.0])

        self.n_points = 20


        self.dphi_max = np.sqrt(2*g/l)
        self.dy_max = 2*np.sqrt(2*mass*g*l/Mass)

        interval_dy = torch.linspace(-self.dy_max, self.dy_max, self.n_points)
        interval_phi = torch.linspace(-np.pi, np.pi, self.n_points)
        interval_dphi = torch.linspace(-self.dphi_max, self.dphi_max, self.n_points)
        # interval_u = torch.linspace(-10, 10, n_points)
        grid_dy, grid_phi, grid_dphi = torch.meshgrid(
            interval_dy,
            interval_phi,
            interval_dphi
        )
        self.grid = torch.cat([
            grid_dy.reshape(-1, 1),
            torch.cos(grid_phi.reshape(-1, 1)),
            torch.sin(grid_phi.reshape(-1, 1)),
            grid_dphi.reshape(-1, 1)
            # grid_u.reshape(-1, 1),
        ], 1)

    def acceleration_x(self, d_y, c_phi, s_phi, d_phi, tensor=False):

        friction_phi = - self.alpha * d_phi
        friction_y = - self.beta * d_y

        dd_phi = self.Mass*self.g*s_phi - c_phi * \
            (self.mass*self.l*d_phi**2*s_phi + friction_y) + \
            self.Mass*friction_phi/(self.mass*self.l)
        dd_phi /= self.Mass*self.l - self.mass*self.l*c_phi**2

        dd_y = self.mass*self.l*s_phi*d_phi**2 + friction_y + \
            friction_phi*c_phi/self.l - self.mass*self.g*s_phi*c_phi
        dd_y /= self.Mass - self.mass * c_phi**2

        return dd_y, dd_phi

    def acceleration_u(self, c_phi, u):
        dd_y = u / (self.Mass - self.mass * c_phi**2)
        dd_phi = - c_phi*u / (self.Mass*self.l - self.mass*self.l*c_phi**2)

        return dd_y, dd_phi
    
    def get_B(self, x):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        B = np.array([
            [0],
            [1/(self.Mass - self.mass * c_phi**2)],
            [0],
            [- c_phi / (self.Mass*self.l - self.mass*self.l*c_phi**2)]
        ])
        return B


    def dynamics(self, x, u):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        dx = np.zeros_like(x)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        dd_y, dd_phi = self.acceleration_x(d_y, c_phi, s_phi, d_phi)
        dd_y_u, dd_phi_u = self.acceleration_u(c_phi, u)
        dd_y += dd_y_u
        dd_phi += dd_phi_u
        dx[0] = d_y
        dx[1] = dd_y
        dx[2] = d_phi
        dx[3] = dd_phi
        return dx
    
    def d_dynamics(self, x, u):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        c_phi, s_phi = torch.cos(phi), torch.sin(phi)
        dx = torch.zeros_like(x)
        dd_y, dd_phi = self.acceleration_x(d_y, c_phi, s_phi, d_phi)
        dd_y_u, dd_phi_u = self.acceleration_u(c_phi, u.squeeze())
        dd_y += dd_y_u
        dd_phi += dd_phi_u
        dx[0] = d_y
        dx[1] = dd_y
        dx[2] = d_phi
        dx[3] = dd_phi
        return dx


    def f_star(self, zeta):
        d_y, c_phi, s_phi, d_phi = zeta[:, 0], zeta[:, 1], zeta[:, 2], zeta[:, 3]
        dd_y, dd_phi = self.acceleration_x(d_y, c_phi, s_phi, d_phi, tensor=True)
        dd = torch.zeros_like(zeta[:, :2])
        # dx[:, 0] = z[:, 1]
        # dx[:, 2] = z[:, 1]
        dd[:, 0] = dd_y
        dd[:, 1] = dd_phi
        return dd

    def step_cost(self, x, u):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        c_phi, s_phi = torch.cos(phi), torch.sin(phi)
        c = 100*y**2+100*(1-c_phi)**2 + 0.1*s_phi**2 + 0.1*d_y**2 + 0.1*d_phi**2 + 0.1*u**2
        # print(f'x = {x}, u={u}, c={c}')
        return c

    def test_error(self, model, x, u, plot, t=0):
        truth = self.f_star(self.grid)
        loss_function = nn.MSELoss()
        predictions = model.net(self.grid.clone()).squeeze()
        # # print(f'prediction {predictions.shape} target {truth.shape} ')
        loss = loss_function(predictions, truth)
        # loss = torch.linalg.norm(self.A_star-model.a_net[0].weight)
        if plot and t % 5 == 0:
            self.plot(x, u, t)
            # plot_portrait(model.forward_x)
            plt.pause(0.1)
            plt.close()
        # print(f'loss = {loss}')
        # print(x)
        return loss

    def plot(self, x, u, t):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        push = 0.7*np.sign(np.mean(u))
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        plt.arrow(y, 0, push, 0, color='red', head_width=0.1, alpha=0.5)
        plt.plot([y-self.l/2, y+self.l/2], [0, 0], color='black')
        plt.plot([y, y+self.l*s_phi], [0, self.l*c_phi], color='blue')
        plt.xlim((y-2*self.l, y+2*self.l))
        plt.ylim((-2*self.l, 2*self.l))
        plt.gca().set_aspect('equal', adjustable='box')

class RlCartpole(Cartpole):
    
    def __init__(self, dt, sigma, alpha, beta):
        gamma = 10.0
        g = 9.8
        mass = 0.1
        Mass = 1.0
        l = 1.0
        super().__init__(dt, sigma, gamma, g, mass, Mass, l, alpha, beta)

        interval_dy = torch.linspace(-self.dy_max, self.dy_max, self.n_points)
        interval_phi = torch.linspace(-np.pi, np.pi, self.n_points)
        interval_dphi = torch.linspace(-self.dphi_max, self.dphi_max, self.n_points)
        interval_u = torch.linspace(-self.gamma, self.gamma, self.n_points)
        grid_dy, grid_phi, grid_dphi, grid_u = torch.meshgrid(
            interval_dy,
            interval_phi,
            interval_dphi,
            interval_u
        )
        self.grid = torch.cat([
            grid_dy.reshape(-1, 1),
            torch.cos(grid_phi.reshape(-1, 1)),
            torch.sin(grid_phi.reshape(-1, 1)),
            grid_dphi.reshape(-1, 1),
            # grid_u.reshape(-1, 1),
        ], 1)

    def acc(self, d_y, c_phi, s_phi, d_phi, u):
        length = self.l / 2
        friction_phi = - self.alpha * d_phi
        friction_y = - self.beta * d_y
        polemass_length = self.mass * length
        temp = (
            u + polemass_length * d_phi**2 * s_phi + friction_y
        ) / self.total_mass
        phiacc = (self.g * s_phi - c_phi * temp - friction_phi/polemass_length) / (
            length * (4.0 / 3.0 - self.mass *
                      c_phi**2 / self.total_mass)
        )
        yacc = temp - polemass_length * phiacc * c_phi / self.total_mass
        return yacc, phiacc

    def dynamics(self, x, u):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        x_dot = np.zeros_like(x)
        yacc, phiacc = self.acc(d_y, c_phi, s_phi, d_phi, u)
        x_dot[0] = d_y
        x_dot[1] = yacc
        x_dot[2] = d_phi
        x_dot[3] = phiacc
        return x_dot

    def f_star(self, zeta):
        d_y, c_phi, s_phi, d_phi = zeta[:,
                                        0], zeta[:, 1], zeta[:, 2], zeta[:, 3]
        u = torch.zeros_like(c_phi)
        yacc, phiacc = self.acc(d_y, c_phi, s_phi, d_phi, u)
        acc = torch.stack((yacc, phiacc), dim=1)
        return acc

    def test_error(self, model, x, u, plot, t=0):
        truth = self.f_star(self.grid)
        loss_function = nn.MSELoss()
        predictions = model.net(self.grid.clone()).squeeze()
        # print(f'prediction {predictions.shape} target {truth.shape} ')
        loss = loss_function(predictions, truth)
        # loss = torch.linalg.norm(self.A_star-model.a_net[0].weight)
        if plot:
            self.plot(x, u, t)
            # plot_portrait(model.forward_x)
            plt.pause(0.1)
            plt.close()
        # print(f'loss = {loss}')
        # print(x)
        return loss


class GymCartpole(RlCartpole):

    def __init__(self, dt=0.02):
        sigma = 0
        alpha = 0.0
        beta = 0.0

        super().__init__(dt, sigma, alpha, beta)

class DmCartpole(RlCartpole):

    def __init__(self, dt=0.02):
        sigma = 0
        alpha = 2e-6
        beta = 5e-4

        super().__init__(dt, sigma, alpha, beta)

        # interval_dy = torch.linspace(-self.dy_max, self.dy_max, self.n_points)
        # interval_phi = torch.linspace(-np.pi, np.pi, self.n_points)
        # interval_dphi = torch.linspace(-self.dphi_max, self.dphi_max, self.n_points)
        # interval_u = torch.linspace(-self.gamma, self.gamma, self.n_points)
        # grid_dy, grid_phi, grid_dphi, grid_u = torch.meshgrid(
        #     interval_dy,
        #     interval_phi,
        #     interval_dphi,
        #     interval_u
        # )
        # self.grid = torch.cat([
        #     grid_dy.reshape(-1, 1),
        #     torch.cos(grid_phi.reshape(-1, 1)),
        #     torch.sin(grid_phi.reshape(-1, 1)),
        #     grid_dphi.reshape(-1, 1),
        #     # grid_u.reshape(-1, 1),
        # ], 1)



    # def get_B(self, x):
    #     y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    #     c_phi, s_phi = np.cos(phi), np.sin(phi)
    #     B_phi = -c_phi / \
    #         (self.l*((4/3)*self.total_mass - self.mass*c_phi**2))

    #     B = np.array([
    #         [0],
    #         [(1-B_phi*c_phi*self.mass*self.l)/self.total_mass ],
    #         [0],
    #         [B_phi],
    #     ])
    #     return B



        
class DampedCartpole(Cartpole):

    def __init__(self, dt=0.02):

        mass, Mass, l = 1.0, 2.0, 1.0
        g = 9.8
        alpha, beta = 0.2, 0.5
        self.period = 2*np.pi / np.sqrt(g/l)
        dt = 1e-2 * self.period
        sigma = 0.01
        gamma = 2

        super().__init__(dt, sigma, gamma, g, mass, Mass, l, alpha, beta)





def sign(x, tensor=False):
    if tensor:
        return torch.sign(x)
    return np.sign(x)


def acceleration_x(d_y, c_phi, s_phi, d_phi, tensor=False):
    friction_phi = - alpha * d_phi
    # friction_y = -beta * Mass * g * sign(d_y, tensor=tensor)
    friction_y = -beta * d_y

    dd_phi = Mass*g*s_phi - c_phi * \
        (mass*l*d_phi**2*s_phi + friction_y) + Mass*friction_phi/(mass*l)
    dd_phi /= Mass*l - mass*l*c_phi**2

    vertical = mass*l*(dd_phi*s_phi+d_phi**2*c_phi) - Mass*g
    # friction_y = -beta * sign(d_y, tensor=tensor) * vertical * sign(vertical, tensor=tensor)
    dd_y = mass*l*s_phi*d_phi**2 + friction_y + \
        friction_phi*c_phi/l - mass*g*s_phi*c_phi
    dd_y /= Mass - mass * c_phi**2

    return dd_y, dd_phi


def acceleration_u(c_phi, u):
    dd_y = u / (Mass - mass * c_phi**2)
    dd_phi = - c_phi*u / (Mass*l - mass*l*c_phi**2)

    return dd_y, dd_phi


def dynamics(x, u):
    y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    x_dot = np.zeros_like(x)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    dd_y, dd_phi = acceleration_x(d_y, c_phi, s_phi, d_phi)
    dd_y_u, dd_phi_u = acceleration_u(c_phi, u)
    dd_y += dd_y_u
    dd_phi += dd_phi_u
    # noise = sigma * np.random.randn(d)
    x_dot[0] = d_y
    x_dot[1] = dd_y
    x_dot[2] = d_phi
    x_dot[3] = dd_phi
    return x_dot


def f_star(z):
    d_y, c_phi, s_phi, d_phi = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    dd_y, dd_phi = acceleration_x(d_y, c_phi, s_phi, d_phi, tensor=True)
    dd = torch.zeros_like(z[:, :2])
    # dx[:, 0] = z[:, 1]
    # dx[:, 2] = z[:, 1]
    dd[:, 0] = dd_y
    dd[:, 1] = dd_phi
    return dd




def test_error(model, x, u, plot, t=0):
    truth = f_star(grid)
    loss_function = nn.MSELoss()
    predictions = model.net(grid.clone()).squeeze()
    # # # print(f'prediction {predictions.shape} target {truth.shape} ')
    loss = loss_function(predictions, truth)
    if plot and t % 10 == 0:
        plot_cartpole(x, u)
        # plot_portrait(model.net)
        plt.pause(0.1)
        plt.close()
    # print(f'loss = {loss}')
    return loss


def plot_cartpole(x, u):
    y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    push = 0.7*np.sign(np.mean(u))
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    plt.arrow(y, 0, push, 0, color='red', head_width=0.1, alpha=0.5)
    plt.plot([y-l/2, y+l/2], [0, 0], color='black')
    plt.plot([y, y+l*s_phi], [0, l*c_phi], color='blue')
    plt.xlim((y-2*l, y+2*l))
    plt.ylim((-2*l, 2*l))
    plt.gca().set_aspect('equal', adjustable='box')


def plot_portrait(f):
    predictions = f(grid)

    # plt.xlim((-phi_max, phi_max))
    # plt.ylim((-dphi_max, dphi_max))
    # vectors = predictions.shape(n_points, n_points, n_points, n_points, 2)
    vector_x = predictions[:, 0].reshape(n_points, n_points).detach().numpy()
    vector_y = predictions[:, 1].reshape(n_points, n_points).detach().numpy()
    magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
    linewidth = magnitude / magnitude.max()
    plt.streamplot(
        grid_dy.numpy().T,
        grid_dphi.numpy().T,
        vector_x.T,
        vector_y.T,
        color='black',
        linewidth=linewidth)
