import torch
import torch.nn as nn

import environments.cartpole as cartpole


class Partial(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            # nn.Linear(8, 8),
            # nn.Tanh(),
            nn.Linear(8, 2)
        )
        self.d, self.m = environment.d, environment.m
        self.get_B = environment.get_B
        self.acceleration_u = environment.acceleration_u
        self.acc_u = environment.acc_u
        self.lr = 0.01


    def transform(self, x):
        y, d_y, phi, d_phi = torch.unbind(x, dim=1)
        # zeta = z[:, :self.d].clone()
        # zeta[:, 0] = z[:, 1]
        # phi = z[:, 2]
        cphi = torch.cos(phi)
        sphi = torch.sin(phi)
        zeta = torch.stack((d_y, cphi, sphi, d_phi), dim=1)
        return zeta
    
    def predict(self, zeta):
        return self.net(zeta)

    def forward(self, z):
        y, d_y, phi, d_phi, u = torch.unbind(z, dim=1)
        cphi = torch.cos(phi)
        sphi = torch.sin(phi)
        zeta = torch.stack((d_y, cphi, sphi, d_phi), dim=1)
        prediction = self.predict(zeta)
        dd_y_x, dd_phi_x = torch.unbind(prediction, dim=1)
        dd_y_u, dd_phi_u = self.acc_u(
            d_y, cphi, sphi, d_phi, u, tensor=True)
        # dx[:, 1::2] = acceleration_x
        # dx[:, 1] += dd_y_u.squeeze()
        # dx[:, 3] += dd_phi_u.squeeze()
        # dx[:, 0] = z[:, 1]
        # dx[:, 2] = z[:, 3]
        dd_y, dd_phi = dd_y_x + dd_y_u, dd_phi_x + dd_phi_u
        dx = torch.stack((d_y, dd_y, d_phi, dd_phi), dim=1)

        # dx = self.forward_x(x)
        # dx = self.forward_u(dx, u)
        return dx
        

class FullNeural(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.d, self.m = environment.d, environment.m

        self.net = nn.Sequential(
            nn.Linear(5, 8),
            nn.Tanh(),
            # nn.Linear(8, 8),
            # nn.Tanh(),
            nn.Linear(8, 2)
        )
        # self.B_net = nn.Sequential(
        #     nn.Linear(3, 16),
        #     nn.Tanh(),
        #     # nn.Linear(16, 16),
        #     # nn.Tanh(),
        #     nn.Linear(16, d)
        # )

        self.lr = 0.005

    # def get_B(self, x):
    #     return self.get_B(x.detach().numpy().squeeze())
        # return self.B_net(self.transform(z)[:, :3]).view(d, m)



    # def acceleration_u(self, z):
    #     x = z[:, :self.d]
    #     B = torch.tensor(self.get_B(x.squeeze().detach().numpy()), dtype=torch.float)
    #     u = z[:, self.d:]
    #     return (B@u.T).T
    # def acceleration_u(self, z):
    #     phi = z[:, 2]
    #     u = z[:, -1]
    #     c_phi = torch.cos(phi)
    #     dd_y_u, dd_phi_u = cartpole.acceleration_u(c_phi, u)
    #     acceleration = torch.zeros_like(z[:, :2])
    #     acceleration[:, 0] = dd_y_u
    #     acceleration[:, 1] = dd_phi_u
    #     return acceleration
    def predict(self, zeta_u):
        return self.net(zeta_u)

    def forward(self, z):
        y, d_y, phi, d_phi, u = torch.unbind(z, dim=1)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        zeta = torch.stack((d_y, cphi, sphi, d_phi), dim=1)
        zeta_u = torch.cat((zeta, u.unsqueeze(0)), dim=1)
        # print(zeta)
        prediction = self.predict(zeta_u)
        dd_y, dd_phi = torch.unbind(prediction, dim=1)
        # x = z[:, :self.d]
        # u = z[:, self.d]
        x_dot = torch.stack((d_y, dd_y, d_phi, dd_phi), dim=1)
        return x_dot
