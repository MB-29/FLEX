import torch
import torch.nn as nn

import environments.cartpole as cartpole


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
        self.d, self.m = environment.d, environment.m
        self.get_B = environment.get_B
        self.acceleration_u = environment.acceleration_u

    # def get_B(self, z):
    #     return cartpole.get_B(z[:, :self.d].detach().numpy().squeeze())
        # return self.B_net(self.transform(z)[:, :3]).view(d, m)

    def transform(self, z):
        zeta = z[:, :self.d].clone()
        zeta[:, 0] = z[:, 1]
        phi = z[:, 2]
        zeta[:, 1] = torch.cos(phi)
        zeta[:, 2] = torch.sin(phi)
        return zeta

    def forward(self, z):
        dx = torch.zeros_like(z[:, :self.d])
        x = self.transform(z)
        u = z[:, self.d:]
        c_phi = torch.cos(z[:, 2])
        # x = z[:, :self.d]
        # u = z[:, self.d]
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
        zeta = z[:, :self.d].clone()
        zeta[:, 0] = z[:, 1]
        phi = z[:, 2]
        zeta[:, 1] = torch.cos(phi)
        zeta[:, 2] = torch.sin(phi)
        return zeta

    def forward(self, z):

        batch_size, _ = z.shape
        x = z[:, :self.d]
        u = z[:, self.d:]
        zeta = self.transform(x)

        zeta_u = torch.zeros((batch_size, self.d+self.m))
        zeta_u[:, :self.d] = zeta
        zeta_u[:, self.d:] = u
        dx = self.net(zeta_u)

        return dx


class GymNeural(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.d, self.m = environment.d, environment.m

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

    # def get_B(self, x):
    #     return self.get_B(x.detach().numpy().squeeze())
        # return self.B_net(self.transform(z)[:, :3]).view(d, m)

    def transform(self, z):
        zeta = z[:, :self.d].clone()
        zeta[:, 0] = z[:, 1]
        phi = z[:, 2]
        zeta[:, 1] = torch.cos(phi)
        zeta[:, 2] = torch.sin(phi)
        return zeta

    def acceleration_u(self, z):
        x = z[:, :self.d]
        B = torch.tensor(self.get_B(x.squeeze().detach().numpy()), dtype=torch.float)
        u = z[:, self.d]
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
    def predict(self, zeta):
        return self.net(zeta)

    def forward(self, z):
        zeta = self.transform(z)
        x = z[:, :self.d]
        u = z[:, self.d]
        dx = torch.zeros_like(x)
        acceleration_u = self.acceleration_u(z)
        dx[:, 1::2] = self.predict(zeta)
        dx += acceleration_u

        # dx = self.forward_x(x)
        # dx = self.forward_u(dx, u)
        return dx
