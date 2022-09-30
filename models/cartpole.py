import torch
import torch.nn as nn

import environments.cartpole as cartpole

d, m = cartpole.d, cartpole.m

class Partial(nn.Module):

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
        self.acceleration_u = environment.acceleration_u

    # def get_B(self, z):
    #     return cartpole.get_B(z[:, :d].detach().numpy().squeeze())
        # return self.B_net(self.transform(z)[:, :3]).view(d, m)

    def transform(self, z):
        zeta = z[:, :d].clone()
        zeta[:, 0] = z[:, 1]
        phi = z[:, 2]
        zeta[:, 1] = torch.cos(phi)
        zeta[:, 2] = torch.sin(phi)
        return zeta

    def forward(self, z):
        dx = torch.zeros_like(z[:, :d])
        x = self.transform(z)
        u = z[:, d:]
        c_phi = torch.cos(z[:, 2])
        # x = z[:, :d]
        # u = z[:, d]
        acceleration_x = self.net(x)
        dd_y_u, dd_phi_u = self.acceleration_u(c_phi, u)
        dx[:, 1::2] = acceleration_x
        dx[:, 1] += dd_y_u.squeeze()
        dx[:, 3] += dd_phi_u.squeeze()
        dx[:, 0] = z[:, 1]
        dx[:, 2] = z[:, 3]

        # dx = self.forward_x(x)
        # dx = self.forward_u(dx, u)
        return dx
        
class NeuralModel(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 4),
            nn.Tanh(),
            # nn.Linear(4, 4),
            # nn.Tanh(),
            nn.Linear(4, 4)
        )
        self.period = environment.period

    def transform(self, z):
        zeta = z[:, :d].clone()
        zeta[:, 0] = z[:, 1]
        phi = z[:, 2]
        zeta[:, 1] = torch.cos(phi)
        zeta[:, 2] = torch.sin(phi)
        return zeta

    def forward(self, z):

        batch_size, _ = z.shape
        x = z[:, :d]
        u = z[:, d:]
        zeta = self.transform(x)

        zeta_u = torch.zeros((batch_size, d+m))
        zeta_u[:, :d] = zeta
        zeta_u[:, d:] = u
        dx = self.net(zeta_u)

        return dx
