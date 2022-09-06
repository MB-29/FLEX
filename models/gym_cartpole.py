import torch
import torch.nn as nn

import environments.cartpole as cartpole

d, m = cartpole.d, cartpole.m
dt = cartpole.dt


class NeuralModel(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            # nn.Linear(16, 16),
            # nn.Tanh(),
            nn.Linear(16, 2)
        )
        # self.B_net = nn.Sequential(
        #     nn.Linear(3, 16),
        #     nn.Tanh(),
        #     # nn.Linear(16, 16),
        #     # nn.Tanh(),
        #     nn.Linear(16, d)
        # )
        self.get_B = environment.get_B

    def get_B(self, x):
        return self.get_B(x.detach().numpy().squeeze())
        # return self.B_net(self.transform(z)[:, :3]).view(d, m)


    def transform(self, z):
        z_ = z[:, :d].clone()
        z_[:, 0] = z[:, 1]
        phi = z[:, 2]
        z_[:, 1] = torch.cos(phi)
        z_[:, 2] = torch.sin(phi)
        return z_

    def acceleration_u(self, z):
        B = torch.tensor(self.get_B(z), dtype=torch.float)
        u = z[:, d]
        return B@u
    # def acceleration_u(self, z):
    #     phi = z[:, 2]
    #     u = z[:, -1]
    #     c_phi = torch.cos(phi)
    #     dd_y_u, dd_phi_u = cartpole.acceleration_u(c_phi, u)
    #     acceleration = torch.zeros_like(z[:, :2])
    #     acceleration[:, 0] = dd_y_u
    #     acceleration[:, 1] = dd_phi_u
    #     return acceleration

    def forward(self, z):
        dx = torch.zeros_like(z[:, :d])
        x = self.transform(z)
        # x = z[:, :d]
        # u = z[:, d]
        acceleration_x = self.net(x)
        acceleration_u = self.acceleration_u(z)
        dx[:, 1::2] = acceleration_x
        dx[:, 0] = z[:, 1]
        dx[:, 2] = z[:, 3]
        dx += acceleration_u

        # dx = self.forward_x(x)
        # dx = self.forward_u(dx, u)
        return dx