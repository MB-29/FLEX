import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.period = environment.period
        self.B_star = torch.tensor(environment.B_star, dtype=torch.float)
        self.d, self.m = environment.d, environment.m


    def transform(self, z):
        return z[:, :self.d]

    def forward_x(self, x):
        raise NotImplementedError

    def forward_u(self, dx, u):
        dx += self.B_star @ u
        return dx

    def predictor(self, z):
        zeta = self.transform(z)
        return self.net(zeta).view(-1)

    def forward(self, z):
        x = self.transform(z)
        # x = z[:, :d]
        u = z[:, self.d]

        dx = self.forward_x(x)
        dx = self.forward_u(dx, u)
        return dx

    def forward_x(self, x):
        dx = torch.zeros_like(x)
        dx[:, 0] = x[:, 1]
        zeta = x.clone()
        zeta[:, 0] = torch.sin(x[:, 0])
        # x[:, 1] = torch.sin(x[:, 1])
        dx[:, 1] = self.net(zeta).view(-1)
        return dx

class NeuralModel(Model):

    def __init__(self, environment):
        super().__init__(environment)
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            # nn.Linear(16, 16),
            # nn.Tanh(),
            nn.Linear(16, 1)
        )
        self.lr = 0.005


class LinearNeural(NeuralModel):

    def __init__(self, environment):
        super().__init__(environment)
        self.net = nn.Sequential(
            nn.Linear(2, 1, bias=False),
        )
        self.lr = 0.05
# class FullLinear(Model):
#     def __init__(self, environment):
#         super().__init__(environment)
#         self.theta = nn.parameter.Parameter(torch.zeros(2, dtype=torch.float))
#         self.lr = 0.01
    
#     def forward_x(self, x):
#         dx = torch.zeros_like(x)
#         dx[:, 0] = x[:, 1]
#         dx[:, 1] = self.theta[0] * torch.sin(x[:, 0]) + self.theta[1]*x[:, 1]
#         return dx


class FullNeural(Model):


    def __init__(self, environment):
        super().__init__(environment)

        self.net = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            # nn.Tanh(),
            # nn.Linear(16, 16),
            # nn.Tanh(),
            # nn.Linear(2, 2)
        )
        self.lr = 0.02
    # Maintenir un lr assez grand pour que les points de forts angles prennent de l'importance
    # evaluer l'erreur d'apprentissage dans la region ou le pendule est oriente vers le haut
    # modifier la regle d'apprentissage pour mieux tenir compte de cette region plus interessante ?
    def transform(self, x):
        batch_size, _ = x.shape
        zeta = torch.zeros(batch_size, 3)
        zeta[:, 0] = torch.cos(x[:, 0])
        zeta[:, 1] = torch.sin(x[:, 0])
        zeta[:, 2] = x[:, 1]
        return zeta


    def forward(self, z):
        batch_size, _ = z.shape
        x = z[:, :self.d]
        # dx = torch.zeros_like(x)
        u = z[:, self.d:]
        zeta = self.transform(x)

        zeta_u = torch.zeros((batch_size, self.d+1+self.m))
        zeta_u[:, :self.d+1] = zeta
        zeta_u[:, self.d+1:] = u
        # dx[:, 0] = x[:, 1]
        dx = self.predict(zeta_u)
        # dx[:, 1] = torch.clip(dx[:, 1].clone(), -8.0, 8.0)
        return dx

    def predict(self, zeta_u):
        zeta = zeta_u[:, :-1]
        u = zeta_u[:, -1:]
        # dx = torch.zeros_like(zeta_u[:, :2])
        # dx[:, 0] = zeta_u[:, 2]
        # print(self.net(zeta).shape)
        # dx[:, 1] = self.net(zeta).squeeze() + ((self.B_star @ u.T).T)[:, 1]
        dx = self.net(zeta_u)
        return dx 



class FullLinear(FullNeural):

    def __init__(self, environment):
        super().__init__(environment)
        self.net = nn.Sequential(
            nn.Linear(3, 2, bias=False),
        )
        self.lr = 0.05
        # self.B_star = torch.tensor(environment.B_star)

    def predict(self, zeta_u):
        zeta = zeta_u[:, :-1]
        u = zeta_u[:, -1:]
        dx = self.net(zeta) + (self.B_star @ u.T).T
        # dx[:, 1] = torch.clip(dx[:, 1].clone(), -8.0, 8.0)
        return dx