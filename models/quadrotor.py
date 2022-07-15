import torch
import torch.nn as nn

import environments.quadrotor as quadrotor


d, m = 6, 2
dt = quadrotor.dt

class NeuralModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            # nn.Linear(16, 16),
            # nn.Tanh(),
            nn.Linear(16, 2)
        )

    def transform(self, z):
        return z[:, 1:4:2]

    def get_B(self, X):
        return dt * quadrotor.get_B(X)

    def forward(self, z):
        dx = torch.zeros_like(z[:, :d])
        v = self.transform(z)
        phi = z[:, 4]
        u = z[:, d:]
        c_phi, s_phi, ux, uy = torch.cos(phi), torch.sin(phi), u[:, 0], u[:, 1]
        friction = self.net(v)
        a_x, a_y, a_phi = quadrotor.known_acceleration(c_phi, s_phi, ux, uy)
        dx[:, 0] = z[:, 1]
        dx[:, 2] = z[:, 3]
        dx[:, 4] = z[:, 3]
        dx[:, 1] = a_x
        dx[:, 3] = a_y
        dx[:, 5] = a_phi
        dx[:, 1:4:2] += friction
        return dx
