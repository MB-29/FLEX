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
        y, d_y, cphi, sphi, d_phi = torch.unbind(obs, dim=1)
        phi = torch.atan2(sphi, cphi)
        x = torch.stack((y, d_y, phi, d_phi), dim=1)
        return x

    def __init__(self, dt, sigma, gamma, g, mass, Mass, l, alpha, beta):
        self.x0 = np.array([0.0, 0.0, np.pi, 0.0])
        super().__init__(self.x0, self.d, self.m, dt, sigma, gamma)

        self.g = g
        self.l = l
        self.mass = mass
        self.Mass = Mass
        self.total_mass = Mass + mass
        self.alpha = alpha
        self.beta = beta

        self.omega_2 = g/l
        self.period = 2*np.pi * np.sqrt(l / g)

        self.n_points = 20


        self.dy_max = 10.0
        self.dphi_max = 10.0
        self.x_min = np.array([-np.inf, -self.dy_max, -np.inf, -self.dphi_max])
        self.x_max = np.array([+np.inf, +self.dy_max, +np.inf, +self.dphi_max])

        self.goal_weights = torch.Tensor([10, 0.1,  10., 10., 0.1])
        self.goal_weights = torch.Tensor([4., 0.1,  10., 6, 0.1])
        self.goal_weights = torch.Tensor([100., 0.1,  100., 100., 0.1])
        self.goal_weights = torch.Tensor([1., 0.1,  10., 5.0, 0.1])
        self.goal_weights = torch.Tensor([10., 0.1,  20., 40., 0.1])
        # self.goal_weights = torch.Tensor([1., 0.1,  20., 20., 0.1])
        self.goal_weights_relaxed = torch.Tensor([10., 0.1,  20., 10., 0.1])
        self.goal_state = torch.Tensor([.0, .0, 1., 0., 0.])
        self.R = 0.001

    def acceleration_x(self, d_y, cphi, s_phi, d_phi, tensor=False):

        friction_phi = - self.alpha * d_phi
        friction_y = - self.beta * d_y

        dd_phi = self.Mass*self.g*s_phi - cphi * \
            (self.mass*self.l*d_phi**2*s_phi + friction_y) + \
            self.Mass*friction_phi/(self.mass*self.l)
        dd_phi /= self.Mass*self.l - self.mass*self.l*cphi**2

        dd_y = self.mass*self.l*s_phi*d_phi**2 + friction_y + \
            friction_phi*cphi/self.l - self.mass*self.g*s_phi*cphi
        dd_y /= self.Mass - self.mass * cphi**2

        return dd_y, dd_phi

    def acceleration_u(self, cphi, u):
        dd_y = u / (self.Mass - self.mass * cphi**2)
        dd_phi = - cphi*u / (self.Mass*self.l - self.mass*self.l*cphi**2)

        return dd_y, dd_phi
    
    def get_B(self, x):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        cphi, s_phi = np.cos(phi), np.sin(phi)
        B = np.array([
            [0],
            [1/(self.Mass - self.mass * cphi**2)],
            [0],
            [- cphi / (self.Mass*self.l - self.mass*self.l*cphi**2)]
        ])
        return B


    def dynamics(self, x, u):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        dx = np.zeros_like(x)
        cphi, s_phi = np.cos(phi), np.sin(phi)
        dd_y, dd_phi = self.acceleration_x(d_y, cphi, s_phi, d_phi)
        dd_y_u, dd_phi_u = self.acceleration_u(cphi, u)
        dd_y += dd_y_u
        dd_phi += dd_phi_u
        dx[0] = d_y
        dx[1] = dd_y
        dx[2] = d_phi
        dx[3] = dd_phi
        # dx_bounded = np.clip(dx, self.velocity_bounds[:, 0], self.velocity_bounds[:, 1])
        return dx
    
    def d_dynamics(self, z):
        y, d_y, phi, d_phi, u = z.unbind(dim=1)
        cphi, s_phi = torch.cos(phi), torch.sin(phi)
        dd_y, dd_phi = self.acceleration_x(d_y, cphi, s_phi, d_phi)
        dd_y_u, dd_phi_u = self.acceleration_u(cphi, u.squeeze())
        dd_y += dd_y_u
        dd_phi += dd_phi_u
        dx = torch.stack((d_y, dd_y, d_phi, dd_phi), dim=1)
        return dx

    def plot_system(self, x, u, t):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        push = 0.06*u[0]
        side = np.sign(push)
        cphi, s_phi = np.cos(phi), np.sin(phi)
        plt.arrow(y+side*self.l/2, 0, push, 0, color='red', head_width=0.1, alpha=0.5)
        plt.plot([y-self.l/2, y+self.l/2], [0, 0], color='black')
        plt.plot([y, y+self.l*s_phi], [0, self.l*cphi], color='blue')
        plt.xlim((y-2*self.l, y+2*self.l))
        plt.ylim((-2*self.l, 2*self.l))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f't = {t}')

class RlCartpole(Cartpole):
    
    def __init__(self, dt, sigma, alpha, beta):
        gamma = 10.0
        g = 9.8
        mass = 0.1
        Mass = 1.0
        l = 1.0
        super().__init__(dt, sigma, gamma, g, mass, Mass, l, alpha, beta)


    def acc(self, d_y, cphi, s_phi, d_phi, u, tensor=False):
        sign = np.sign(d_y) if not tensor else torch.sign(d_y)
        length = self.l / 2
        friction_phi = - self.alpha * d_phi
        friction_y = - self.beta * sign
        polemass_length = self.mass * length
        temp = (
            u + polemass_length * d_phi**2 * s_phi + friction_y
        ) / self.total_mass
        phiacc = (self.g * s_phi - cphi * temp + friction_phi/polemass_length) / (
            length * (4.0 / 3.0 - self.mass *
                      cphi**2 / self.total_mass)
        )
        yacc = temp - polemass_length * phiacc * cphi / self.total_mass
        return yacc, phiacc

    def acc_u(self, d_y, cphi, s_phi, d_phi, u, tensor=False):
        u0 = torch.zeros_like(u)
        yacc, phiacc = self.acc(d_y, cphi, s_phi, d_phi, u, tensor)
        yacc0, phiacc0 = self.acc(d_y, cphi, s_phi, d_phi, u0, tensor)
        return yacc-yacc0, phiacc-phiacc0

    def dynamics(self, x, u):
        y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
        cphi, s_phi = np.cos(phi), np.sin(phi)
        dd_y, dd_phi = self.acc(d_y, cphi, s_phi, d_phi, u)
        x_dot = np.array([d_y, dd_y, d_phi, dd_phi], dtype=float)
        # x_dot_bounded = np.clip(
        #     x_dot, self.velocity_bounds[:, 0], self.velocity_bounds[:, 1])
        return x_dot
    
    def d_dynamics(self, z):
        y, d_y, phi, d_phi, u = z.unbind(dim=1)
        cphi, s_phi = torch.cos(phi), torch.sin(phi)
        dd_y, dd_phi = self.acc(d_y, cphi, s_phi, d_phi, u, tensor=True)
        dx = torch.stack((d_y, dd_y, d_phi, dd_phi), dim=1)
        return dx

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


    # def get_B(self, x):
    #     y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
    #     cphi, s_phi = np.cos(phi), np.sin(phi)
    #     B_phi = -cphi / \
    #         (self.l*((4/3)*self.total_mass - self.mass*cphi**2))

    #     B = np.array([
    #         [0],
    #         [(1-B_phi*cphi*self.mass*self.l)/self.total_mass ],
    #         [0],
    #         [B_phi],
    #     ])
    #     return B



        
class DampedCartpole(RlCartpole):

    def __init__(self, dt=0.02):

        alpha, beta = 0.05, 0.5
        sigma = 0.0

        super().__init__(dt, sigma, alpha, beta)
# class DampedCartpole(Cartpole):

#     def __init__(self, dt=0.02):

#         mass, Mass, l = 1.0, 2.0, 1.0
#         g = 9.8
#         alpha, beta = 0.2, 0.5
#         self.period = 2*np.pi / np.sqrt(g/l)
#         dt = 1e-2 * self.period
#         sigma = 0.01
#         gamma = 4

#         super().__init__(dt, sigma, gamma, g, mass, Mass, l, alpha, beta)





# def sign(x, tensor=False):
#     if tensor:
#         return torch.sign(x)
#     return np.sign(x)


# def acceleration_x(d_y, cphi, s_phi, d_phi, tensor=False):
#     friction_phi = - alpha * d_phi
#     # friction_y = -beta * Mass * g * sign(d_y, tensor=tensor)
#     friction_y = -beta * d_y

#     dd_phi = Mass*g*s_phi - cphi * \
#         (mass*l*d_phi**2*s_phi + friction_y) + Mass*friction_phi/(mass*l)
#     dd_phi /= Mass*l - mass*l*cphi**2

#     vertical = mass*l*(dd_phi*s_phi+d_phi**2*cphi) - Mass*g
#     # friction_y = -beta * sign(d_y, tensor=tensor) * vertical * sign(vertical, tensor=tensor)
#     dd_y = mass*l*s_phi*d_phi**2 + friction_y + \
#         friction_phi*cphi/l - mass*g*s_phi*cphi
#     dd_y /= Mass - mass * cphi**2

#     return dd_y, dd_phi


# def acceleration_u(cphi, u):
#     dd_y = u / (Mass - mass * cphi**2)
#     dd_phi = - cphi*u / (Mass*l - mass*l*cphi**2)

#     return dd_y, dd_phi


# def dynamics(x, u):
#     y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
#     x_dot = np.zeros_like(x)
#     cphi, s_phi = np.cos(phi), np.sin(phi)
#     dd_y, dd_phi = acceleration_x(d_y, cphi, s_phi, d_phi)
#     dd_y_u, dd_phi_u = acceleration_u(cphi, u)
#     dd_y += dd_y_u
#     dd_phi += dd_phi_u
#     # noise = sigma * np.random.randn(d)
#     x_dot[0] = d_y
#     x_dot[1] = dd_y
#     x_dot[2] = d_phi
#     x_dot[3] = dd_phi
#     return x_dot


# def f_star(z):
#     d_y, cphi, s_phi, d_phi = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
#     dd_y, dd_phi = acceleration_x(d_y, cphi, s_phi, d_phi, tensor=True)
#     dd = torch.zeros_like(z[:, :2])
#     # dx[:, 0] = z[:, 1]
#     # dx[:, 2] = z[:, 1]
#     dd[:, 0] = dd_y
#     dd[:, 1] = dd_phi
#     return dd




# def test_error(model, x, u, plot, t=0):
#     truth = f_star(grid)
#     loss_function = nn.MSELoss()
#     predictions = model.net(grid.clone()).squeeze()
#     # # # print(f'prediction {predictions.shape} target {truth.shape} ')
#     loss = loss_function(predictions, truth)
#     if plot and t % 10 == 0:
#         plot_system(x, u)
#         # plot_portrait(model.net)
#         plt.pause(0.1)
#         plt.close()
#     # print(f'loss = {loss}')
#     return loss


# def plot_system(x, u):
#     y, d_y, phi, d_phi = x[0], x[1], x[2], x[3]
#     push = 0.7*np.sign(np.mean(u))
#     cphi, s_phi = np.cos(phi), np.sin(phi)
#     plt.arrow(y, 0, push, 0, color='red', head_width=0.1, alpha=0.5)
#     plt.plot([y-l/2, y+l/2], [0, 0], color='black')
#     plt.plot([y, y+l*s_phi], [0, l*cphi], color='blue')
#     plt.xlim((y-2*l, y+2*l))
#     plt.ylim((-2*l, 2*l))
#     plt.gca().set_aspect('equal', adjustable='box')


# def plot_portrait(f):
#     predictions = f(grid)

#     # plt.xlim((-phi_max, phi_max))
#     # plt.ylim((-dphi_max, dphi_max))
#     # vectors = predictions.shape(n_points, n_points, n_points, n_points, 2)
#     vector_x = predictions[:, 0].reshape(n_points, n_points).detach().numpy()
#     vector_y = predictions[:, 1].reshape(n_points, n_points).detach().numpy()
#     magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
#     linewidth = magnitude / magnitude.max()
#     plt.streamplot(
#         grid_dy.numpy().T,
#         grid_dphi.numpy().T,
#         vector_x.T,
#         vector_y.T,
#         color='black',
#         linewidth=linewidth)
