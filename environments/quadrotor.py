import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

from environments.environment import Environment

d, m = 6, 2

class Quadrotor(Environment):

    def __init__(self, dt, sigma, gamma, mass, g, I, r, rho):
        super().__init__(d, m, dt, sigma, gamma)
        self.mass = mass
        self.g = g
        self.I = I
        self.rho = rho
        self.r = r

        self.x0 = np.array([0, 0, 0.0, 0, 0, 0])

        n_points = 20

        v_max = 2*np.sqrt(gamma)
        interval_v = torch.linspace(-v_max, v_max, n_points)
        grid_vx, grid_vy = torch.meshgrid(
            interval_v,
            interval_v,
        )
        self.grid = torch.cat([
            grid_vx.reshape(-1, 1),
            grid_vy.reshape(-1, 1)
            # grid_u.reshape(-1, 1),
        ], 1)

    def get_B(self, X):
        x, v_x, y, v_y, phi, d_phi = X[0], X[1], X[2], X[3], X[4], X[5]
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        B = np.array([
            [0, 0],
            [-s_phi / self.mass, -s_phi / self.mass],
            [0, 0],
            [c_phi / self.mass, c_phi / self.mass],
            [0, 0],
            [self.r/self.I, -self.r/self.I]
        ])
        return B

    def known_acceleration(self, c_phi, s_phi, u1, u2):
        a_x = -s_phi*(u1 + u2) / self.mass
        a_y = (c_phi*(u1 + u2) - self.mass*self.g)/self.mass
        a_phi = self.r*(u1 - u2) / self.I
        return a_x, a_y, a_phi


    def dynamics(self, X, u):
        x, v_x, y, v_y, phi, d_phi = X[0], X[1], X[2], X[3], X[4], X[5]
        X_dot = np.zeros_like(X)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        f_x = - (self.rho/self.mass) * np.abs(v_x) * v_x
        f_y = - (self.rho/self.mass) * np.abs(v_y) * v_y
        a_x, a_y, a_phi = self.known_acceleration(c_phi, s_phi, u[0], u[1])
        # print(f'phi = {phi%(2*np.pi)}')
        # print(f'a_x = {a_x}')
        X_dot[0] = X[1]
        X_dot[1] = a_x + f_x
        X_dot[2] = X[3] 
        X_dot[3] = a_y + f_y
        X_dot[4] = X[5] 
        X_dot[5] = a_phi
        return X_dot

    def f_star(self, v):
        return -(self.rho/self.mass)*torch.abs(v)*v


    def test_error(self, model, X, u, plot, t=0):
        loss_function = nn.MSELoss()
        truth = self.f_star(self.grid)
        predictions = model.net(self.grid.clone()).squeeze()
        # # # print(f'prediction {predictions.shape} target {truth.shape} ')
        loss = loss_function(predictions, truth)
        if plot and t%10==0:
            plt.subplot(121)
            self.plot(X, u)
            plt.subplot(122)
            # plot_portrait(g_star)
            # plot_portrait(model.net)
            plt.pause(0.1)
            plt.close()
        # print(f'loss = {loss}')
        return loss

    def plot(self, X, u):
        x, v_x, y, v_y, phi, d_phi = X[0], X[1], X[2], X[3], X[4], X[5]
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        l0 = u[0]/self.gamma
        l1 = u[1]/self.gamma
        plt.arrow(x-self.r*c_phi, y-self.r*s_phi, -l0*s_phi, l0*c_phi,
                color='red', head_width=0.1, alpha=0.5)
        plt.arrow(x+self.r*c_phi, y+self.r*s_phi, -l1*s_phi, l1*c_phi,
                color='red', head_width=0.1, alpha=0.5)
        plt.plot([x-self.r*c_phi, x+self.r*c_phi],
                [y-self.r*s_phi, y+self.r*s_phi], color='blue')
        # plt.xlim((x-2*r, x+2*r))
        # plt.ylim((y-2*r, y+2*r))
        plt.xlim((-10, 10))
        plt.ylim((-10, 10))
        plt.gca().set_aspect('equal', adjustable='box')


class DefaultQuadrotor(Quadrotor):

    def __init__(self, dt=0.1):
        mass, I, r = 2.0, 10, 1.0
        g = 0.0
        sigma = 0.0
        gamma = 10
        rho = 0.2
        super().__init__(dt, sigma, gamma, mass, g ,I, r, rho)




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
        grid_vx.numpy().T,
        grid_vy.numpy().T,
        vector_x.T,
        vector_y.T,
        color='black',
        linewidth=linewidth
        )
