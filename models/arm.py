import torch
import torch.nn as nn

import environments.arm as arm

d, m = arm.d, arm.m
dt = arm.dt


class NeuralModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.Tanh(),
            # nn.Linear(16, 16),
            # nn.Tanh(),
            nn.Linear(16, 2)
        )

    def get_B(self, x):
        return dt*arm.get_B(x)

    def transform(self, z):
        batch_size, _ = z.shape
        zeta = torch.zeros(batch_size, 6)
        zeta[:, 0] = torch.cos(z[:, 0])
        zeta[:, 1] = torch.cos(z[:, 2])
        zeta[:, 2] = torch.sin(z[:, 0])
        zeta[:, 3] = torch.sin(z[:, 2])
        zeta[:, 4] = z[:, 1]
        zeta[:, 4] = z[:, 3]
        return zeta

    def acceleration_u(self, z):
        phi1, phi2 = z[:, 0], z[:, 2]
        u = z[:, d:]
        c_dphi = torch.cos(phi1-phi2)
        s_dphi = torch.sin(phi1-phi2)
        dd_y_u, dd_phi_u = arm.acceleration_u(u[:, 0], u[:, 1], c_dphi, s_dphi)
        acceleration = torch.zeros_like(z[:, :2])
        acceleration[:, 0] = dd_y_u
        acceleration[:, 1] = dd_phi_u
        return acceleration

    def forward(self, z):
        dx = torch.zeros_like(z[:, :d])
        x = self.transform(z)
        # x = z[:, :d]
        # u = z[:, d]
        acceleration_x = self.net(x)
        acceleration_u = self.acceleration_u(z)
        acceleration = acceleration_x
        acceleration += acceleration_u
        dx[:, 1::2] = acceleration
        dx[:, 0] = z[:, 1]
        dx[:, 2] = z[:, 3]

        # dx = self.forward_x(x)
        # dx = self.forward_u(dx, u)
        return dx
