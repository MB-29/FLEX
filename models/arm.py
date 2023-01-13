import torch
import torch.nn as nn

from evaluation.arm import XGrid

class NeuralA(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.d, self.m = environment.d, environment.m
        # self.get_B = environment.get_B
        self.t_period = environment.period/environment.dt
        self.acceleration_u = environment.acceleration_u
        self.acceleration_x = environment.acceleration_x

        self.evaluation  = XGrid(environment)

        # self.evaluation = XGrid(environment)
        self.net = nn.Sequential(
            nn.Linear(6, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            # nn.Tanh(),
            # nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
        )
        self.lr = 0.05

    # def get_B(self, x):
    #     return dt*arm.get_B(x)

    # def transform(self, z):
    #     batch_size, _ = z.shape
    #     zeta = torch.zeros(batch_size, 6)
    #     zeta[:, 0] = torch.cos(z[:, 0])
    #     zeta[:, 1] = torch.cos(z[:, 2])
    #     zeta[:, 2] = torch.sin(z[:, 0])
    #     zeta[:, 3] = torch.sin(z[:, 2])
    #     zeta[:, 4] = z[:, 1]
    #     zeta[:, 4] = z[:, 3]
    #     return zeta

    # def acceleration_u(self, z):
    #     phi1, phi2 = z[:, 0], z[:, 2]
    #     u = z[:, self.d:]
    #     c_dphi = torch.cos(phi1-phi2)
    #     s_dphi = torch.sin(phi1-phi2)
    #     dd_y_u, dd_phi_u = arm.acceleration_u(u[:, 0], u[:, 1], c_dphi, s_dphi)
    #     acceleration = torch.zeros_like(z[:, :2])
    #     acceleration[:, 0] = dd_y_u
    #     acceleration[:, 1] = dd_phi_u
    #     return acceleration

    def forward(self, z):
        phi1, d_phi1, phi2, d_phi2, u1, u2 = z.unbind(dim=1)
        cphi1, sphi1 = torch.cos(phi1), torch.sin(phi1)
        cphi2, sphi2 = torch.cos(phi2), torch.sin(phi2)
        delta = phi1 - phi2
        cdelta, sdelta = torch.cos(delta), torch.sin(delta)
        # x = z[:, :self.d]
        # u = z[:, d]
        obs = torch.stack((cphi1, sphi1, cphi2, sphi2, d_phi1, d_phi2), dim=1)
        # print(obs)

        # dd_phi1, dd_phi2 = torch.unbind(prediction, dim=1)
        prediction = self.predict(obs)

        cdelta = cphi1*cphi2 + sphi1*sphi2
        sdelta = sphi1*cphi2 - sphi2*cphi1
        s2delta = 2*sdelta*cdelta
        a1, a2 = self.acceleration_x(
            cphi1, cphi2, sphi1, cdelta, sdelta, s2delta, d_phi1, d_phi2)
        au1, au2 = self.acceleration_u(u1, u2, cdelta, sdelta)

        # print(f'a = {a1}')
        # print(f'au = {au1}')
            
        dd_x = torch.stack((a1, a2), dim=1) 
        dd = torch.stack((au1, au2), dim=1) + prediction
        # dd = torch.stack((a1+au1, a2+au2), dim=1) + prediction
        # if z.shape[0] == 1:
        #     print(f'prediction error {prediction - dd_x}')
            # print(f'a = {a1}, {a2}')
            # print(f'f star = {self.evaluation.f_star(obs)}')
        
        # print(f'diff 0 = {a1-prediction[:, 0]}')
        # print(f'diff 1 = {a2-prediction[:, 1]}')
        # dd = prediction

        dd_phi1, dd_phi2 = torch.unbind(dd, dim=1)
        x_dot = torch.stack((d_phi1, dd_phi1, d_phi2, dd_phi2), dim=1)
        
        # dx = self.forward_x(x)
        # dx = self.forward_u(dx, u)
        return x_dot
    
    def predict(self, obs):
        # print(f'obs = {obs}')
        # cphi1, sphi1, cphi2, sphi2, d_phi1, d_phi2 = torch.unbind(obs, dim=1)
       
        # return 0.9*dd + self.net(obs)
        return self.net(obs)
