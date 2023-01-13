import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt

from environments.environment import Environment




x0 = np.array([0.0, 0.0, 0.0, 0.0])

class Arm(Environment):
    d, m = 4, 2

    def __init__(self, dt, sigma, gamma, g, m1, m2, l1, l2, alpha):
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0])
        super().__init__(self.x0, self.d, self.m, dt, sigma, gamma)

        self.g = g
        self.alpha = alpha
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.period = 2*np.pi * np.sqrt(l1 / g)

        self.phi_max = np.pi/5
        self.dphi_max = 2*np.sqrt(2*g/(l1+l2))
        self.dphi_max = 3.0

        self.x_min = np.array([-np.inf, -self.dphi_max, -np.inf, -self.dphi_max])
        self.x_max = np.array([+np.inf, +self.dphi_max, +np.inf, +self.dphi_max])


    def acceleration_x(self, cphi1, cphi2, sphi1, cdelta, sdelta, s2delta, d_phi1, d_phi2):
        friction1 = - self.alpha*d_phi1
        friction2 = - self.alpha*d_phi2

        a_phi1 = self.m2*self.l1*d_phi1**2*s2delta
        a_phi1 += 2*self.m2*self.l2*d_phi2**2*sdelta
        a_phi1 += 2*self.g*self.m2*cphi2*sdelta
        a_phi1 += 2*self.g*self.m1*sphi1
        a_phi1 += -2*friction1 + 2*friction2*cdelta
        a_phi1 /= -2*self.l1*(self.m1+self.m2*sdelta**2)

        a_phi2 = self.m2*self.l2*d_phi2**2*s2delta
        a_phi2 += 2*(self.m1+self.m2)*self.l1*d_phi1**2*sdelta
        a_phi2 += 2*self.g*(self.m1+self.m2)*cphi1*sdelta
        a_phi2 += 2*((self.m1+self.m2)/self.m2)*friction2 - 2*friction1*cdelta
        a_phi2 /= 2*self.l2*(self.m1+self.m2*sdelta**2)

        return a_phi1, a_phi2
        
    def acceleration_u(self, u1, u2, c_dphi, s_dphi):
        a_phi1 = -2*u1
        a_phi1 /= -2*self.l1*(self.m1+self.m2*s_dphi**2)
        a_phi2 = 2*((self.m1+self.m2)/self.m2)*u2 - 2*u1*c_dphi
        a_phi2 /= 2*self.l2*(self.m1+self.m2*s_dphi**2)

        return a_phi1, a_phi2

    # def get_B(self, x):
    #     phi1, d_phi1, phi2, d_phi2 = x
    #     c_phi1, s_phi1 = np.cos(phi1), np.sin(phi1)
    #     c_phi2, s_phi2 = np.cos(phi2), np.sin(phi2)
    #     c_dphi = np.cos(phi1-phi2)
    #     s_dphi = np.sin(phi1-phi2)
    #     M = (self.m1 + self.m2 * s_dphi**2)
    #     B = np.array([
    #         [0, 0],
    #         [1 / (self.l1*M), -c_dphi / (self.l1*M)],
    #         [0, 0],
    #         [-c_dphi / (self.l2*M), ((self.m1+self.m2)/self.m2) / (self.l2*M)]
    #     ])
    #     return B

    def dynamics(self, x, u):
        phi1, d_phi1, phi2, d_phi2 = x
        cphi1, sphi1 = np.cos(phi1), np.sin(phi1)
        cphi2, sphi2 = np.cos(phi2), np.sin(phi2)
        delta = phi1 - phi2
        cdelta, sdelta = np.cos(delta), np.sin(delta)
        s2delta = np.sin(2*(delta))
        x_dot = np.zeros_like(x)
        x_dot[0] = x[1]
        x_dot[2] = x[3]
        a_phi1, a_phi2 = self.acceleration_x(
            cphi1, cphi2, sphi1, cdelta, sdelta, s2delta, d_phi1, d_phi2)
        a_phi1_u, a_phi2_u = self.acceleration_u(u[0], u[1], cdelta, sdelta)
        # print(a_phi1_u)
        # print(a_phi2_u)
        x_dot[1] = a_phi1 + a_phi1_u
        x_dot[3] = a_phi2 + a_phi2_u
        return x_dot


    def plot_system(self, x, u, t):
        phi1, d_phi1, phi2, d_phi2 = x[0], x[1], x[2], x[3]
        # push = 0.7*np.sign(np.mean(u))
        c_phi1, s_phi1 = np.cos(phi1), np.sin(phi1)
        c_phi2, s_phi2 = np.cos(phi2), np.sin(phi2)

        plt.arrow(
            self.l1*s_phi1,
            -self.l1*c_phi1,
            0.05*u[0]*c_phi1,
            0.05*u[0]*s_phi1,
            color='red',
            head_width=0.1,
            head_length=0.01*abs(u[0]),
            alpha=0.5)
        plt.arrow(
            self.l1*s_phi1 + self.l2*s_phi2,
            -self.l1*c_phi1 - self.l2*c_phi2,
            0.05*u[1]*c_phi2,
            0.05*u[1]*s_phi2,
            color='red',
            head_width=0.1,
            head_length=0.01*abs(u[1]),
            alpha=0.5)

        plt.plot([0, self.l1*s_phi1], [0, -self.l1*c_phi1], color='black')
        plt.plot([self.l1*s_phi1, self.l1*s_phi1 + self.l2*s_phi2],
                [-self.l1*c_phi1, -self.l1*c_phi1-self.l2*c_phi2], color='blue')
        total_length=self.l1+self.l2
        plt.xlim((-(1.5*total_length), 1.5*total_length))
        plt.ylim((-(1.5*total_length), 1.5*total_length))
        plt.gca().set_aspect('equal', adjustable='box')

class DampedArm(Arm):

    def __init__(self):
        m1, m2 = 1, 1
        l1, l2 = 1, 1
        g = 9.8
        alpha = 2.0

        period = 2*np.pi * np.sqrt(l1 / g)
        dt = 2*1e-2 * period
        sigma = 0.0
        gamma = 5.0
        super().__init__(dt, sigma, gamma, g, m1, m2, l1, l2, alpha)
