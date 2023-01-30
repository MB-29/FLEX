import torch
import torch.nn as nn

from evaluation.arm import XGrid

class NeuralA(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.d, self.m = environment.d, environment.m
        self.t_period = environment.period/environment.dt
        self.acceleration_u = environment.acceleration_u

        self.evaluation  = XGrid(environment)

        self.net = nn.Sequential(
            nn.Linear(6, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
        )
        self.lr = 0.05


    def forward(self, z):
        phi1, d_phi1, phi2, d_phi2, u1, u2 = z.unbind(dim=1)
        cphi1, sphi1 = torch.cos(phi1), torch.sin(phi1)
        cphi2, sphi2 = torch.cos(phi2), torch.sin(phi2)
        delta = phi1 - phi2
        cdelta, sdelta = torch.cos(delta), torch.sin(delta)
        obs = torch.stack((cphi1, sphi1, cphi2, sphi2, d_phi1, d_phi2), dim=1)

        prediction = self.predict(obs)

        cdelta = cphi1*cphi2 + sphi1*sphi2
        sdelta = sphi1*cphi2 - sphi2*cphi1
        au1, au2 = self.acceleration_u(u1, u2, cdelta, sdelta)

        dd = torch.stack((au1, au2), dim=1) + prediction

        dd_phi1, dd_phi2 = torch.unbind(dd, dim=1)
        x_dot = torch.stack((d_phi1, dd_phi1, d_phi2, dd_phi2), dim=1)
        
        return x_dot
    
    def predict(self, obs):
        return self.net(obs)
