import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

from environments.environment import Environment


class Potential(Environment):
    d, m = 4, 2

    def __init__(self, dt, sigma, gamma, alpha):
        self.x0 = np.array([-0.0, 0.0, -0.0, 0.0])
        self.x0 = np.array([0.8, -0.0, -0.0, 0.0])
        super().__init__(self.x0, self.d, self.m, dt, sigma, gamma)

        self.alpha = alpha
        self.omega = 2*np.pi/1000

        self.v_max = 5.0
        self.q_max = 1.0

        self.x_min = np.array(
            [-self.q_max, -self.v_max, -self.q_max, -self.v_max])
        self.x_max = np.array(
            [+self.q_max, +self.v_max, +self.q_max, +self.v_max])
    
    def force(self, qx, qy, cx, cy, tensor=False):
      
        # self.cx = t/self.q_max
        # self.cy = t/self.q_ma 
        if not tensor:
            relative = np.array([qx-cx, qy-cy])
            norm = np.linalg.norm(relative)
            sq_norm = norm**2
            repulse = np.exp(-sq_norm / 0.02**2)
        else:
            relative = torch.stack((qx-cx, qy-cy), dim=1)
            sq_norm = (relative**2).sum(dim=1).unsqueeze(1)
            norm = torch.sqrt(sq_norm)
            repulse = torch.exp(-sq_norm / 0.02**2)
            # norm = torch.norm(relative, dim=1).unsqueeze(1)
            # print(norm.shape)
        factor = 1/(1+ sq_norm/(0.5**2)) + 1e9*repulse
        return factor * relative / norm 
    def center_position(self, t):
        return 0.9*np.cos(self.omega*t),  0.9*np.sin(self.omega*t)

    def dynamics(self, x, u, t):
        qx, vx, qy, vy = x
        self.cx, self.cy = self.center_position(t)
        friction = - self.alpha * np.array([vx, vy])
        force = self.force(qx, qy, self.cx, self.cy)
        ax, ay = force + friction + u
        # ax, ay = fx + u[0], fy + u[1]
        # print(f'phi = {phi%(2*np.pi)}')
        # print(f'a_x = {a_x}')
        x_dot = np.array([vx, ax, vy, ay])
        return x_dot

    def plot_system(self, x, u, t):
        qx, vx, qy, vy = x
        lx = 0.2*u[0]/self.gamma
        ly = 0.2*u[1]/self.gamma
        plt.scatter(qx, qy, marker='o')
        plt.arrow(qx, qy, lx, ly,
                  color='red', head_width=0.01, alpha=0.5)
        plt.scatter(self.cx, self.cy, marker='o')
        # plt.xlim((x-2*r, x+2*r))
        # plt.ylim((y-2*r, y+2*r))
        plt.title(f't = {t}')
        plt.xlim((-self.q_max, self.q_max))
        plt.ylim((-self.q_max, self.q_max))
        plt.gca().set_aspect('equal', adjustable='box')
    

class DefaultPotential(Potential):

    def __init__(self):
        dt = 0.01
        sigma = 1e-6
        # sigma = 0
        gamma = 50.
        alpha = 100. 
        super().__init__(dt, sigma, gamma, alpha)



