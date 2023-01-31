import numpy as np
import torch
import torch.nn as nn

from evaluation.cartpole import GridA, GridAB

class NeuralA(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
        )
        self.d, self.m = environment.d, environment.m
        self.t_period = environment.period/environment.dt
        self.acceleration_u = environment.acceleration_u
        self.acc_u = environment.acc_u
        self.lr = 0.1

        self.evaluation = GridA(environment)


    def transform(self, x):
        y, d_y, phi, d_phi = torch.unbind(x, dim=1)
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
        dd_y, dd_phi = dd_y_x + dd_y_u, dd_phi_x + dd_phi_u
        dx = torch.stack((d_y, dd_y, d_phi, dd_phi), dim=1)

        return dx
        

class NeuralAB(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.d, self.m = environment.d, environment.m
        self.evaluation = GridAB(environment)
        self.t_period = environment.period / environment.dt

        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 4)
        )

        self.lr = 0.1

    def predict(self, obs_u):
        d_y, cphi, sphi, d_phi, u = torch.unbind(obs_u, dim=1)
        zeta = obs_u[:, :-1]
        vectors = self.net(zeta)
        return vectors[:, :2] + u.unsqueeze(1)*vectors[:, 2:]

    def forward(self, z):
        y, d_y, phi, d_phi, u = torch.unbind(z, dim=1)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        zeta = torch.stack((d_y, cphi, sphi, d_phi), dim=1)
        zeta_u = torch.cat((zeta, u.unsqueeze(1)), dim=1)
        prediction = self.predict(zeta_u)
        dd_y, dd_phi = torch.unbind(prediction, dim=1)
        x_dot = torch.stack((d_y, dd_y, d_phi, dd_phi), dim=1)
        return x_dot


class RFF(nn.Module):
    """Random Fourier Features,
    see [Schultheis et al., 2019, Receding Horizon Curiosity]
    """
    def __init__(self, environment) -> None:
        super().__init__()
        self.linear = True
        self.evaluation = GridAB(environment)

        dy_max = environment.dy_max
        dphi_max = environment.dphi_max
        self.q = 25
        self.P = torch.randn(self.q, 5)
        self.nu = torch.tensor([1., 1., 1., 1., environment.gamma]).unsqueeze(0)
        self.phases = 2*np.pi*(torch.rand(1, self.q) -0.5)
        self.net = nn.Sequential(
            nn.Linear(self.q, 2, bias=False),
        )

    def feature(self, z):
        y, d_y, phi, d_phi, u = torch.unbind(z, dim=1)
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        zeta = torch.stack((d_y, cphi, sphi, d_phi), dim=1)
        obs_u = torch.cat((zeta, u.unsqueeze(0)), dim=1)
        return self.fourier(obs_u)

    def fourier(self, obs_u):
        zeta_u = obs_u / self.nu
        xi = torch.sin((self.P@zeta_u.T).T + self.phases)
        return xi
    def predict(self, obs_u):
        xi = self.fourier(obs_u)
        return self.net(xi)
        
    
    def forward(self, z):
        y, d_y, phi, d_phi, u = torch.unbind(z, dim=1)
        xi = self.feature(z)
        prediction = self.net(xi)
        dd_y, dd_phi = torch.unbind(prediction, dim=1)
        x_dot = torch.stack((d_y, dd_y, d_phi, dd_phi), dim=1)
        return x_dot

