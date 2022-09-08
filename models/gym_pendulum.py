import numpy as np
import torch
import torch.nn as nn

import environments.gym_pendulum as gym_pendulum

d, m = gym_pendulum.d, gym_pendulum.m


class Model(nn.Module):

    def __init__(self, environment):
        super().__init__()

    def transform(self, x):
        batch_size, _ = x.shape
        zeta = torch.zeros(batch_size, 3)
        zeta[:, 0] = torch.cos(x[:, 0])
        zeta[:, 1] = torch.sin(x[:, 0])
        zeta[:, 2] = x[:, 1]
        return zeta

    def forward_x(self, x):
        raise NotImplementedError

    def forward_u(self, dx, u):
        raise NotImplementedError
    
    def model(self, x, u):
        z = torch.zeros(1, d+m)
        z[:, :d] = x
        z[:, d] = u
        return self.forward(z).squeeze()

    # def predictor(self, z):
        # zeta = self.transform(z)
        # return self.net(zeta).view(-1)


class NeuralModel(Model):

    def __init__(self, environment):
        super().__init__(environment)

    
        self.net = nn.Sequential(
            nn.Linear(4, 5),
            nn.Tanh(),
            # nn.Linear(5, 5),
            # nn.Tanh(),
            nn.Linear(5, 2)
        )
    def forward(self, z):
        batch_size, _ = z.shape
        x = z[:, :d]
        u = z[:, d:]
        zeta = self.transform(x)

        zeta_u = torch.zeros((batch_size, d+1+m))
        zeta_u[:, :d+1] = zeta
        zeta_u[:, d+1:] = u
        dx = self.net(zeta_u)

        return dx

# class NeuralModel(Model):

#     def __init__(self):
#         super().__init__()

    
#         self.a_net = nn.Sequential(
#             nn.Linear(3, 16),
#             nn.Tanh(),
#             # nn.Linear(16, 16),
#             # nn.Tanh(),
#             nn.Linear(16, 2)
#         )
    
#         # self.B_net = nn.Sequential(
#         #     nn.Linear(3, 2),
#         # )
#         self.B_net = nn.Sequential(
#             nn.Linear(3, 2),
#             nn.Tanh(),
#             nn.Linear(2, 2),
#             nn.Tanh(),
#             nn.Linear(2, 2)
#         )

#         self.lr = 0.0005

#     def forward(self, z):

#         x = z[:, :d]
#         # x = z[:, :d]

#         dx = self.forward_x(x) + self.forward_u(z)
#         # print(f'dx = {dx}')
#         # dx += self.forward_u(z)
#         return dx
    


#     def forward_x(self, x):

#         dx = self.a_net(self.transform(x))

#         # dx = torch.zeros_like(x)
#         # dx[:, 0] = x[:, 1]
#         # dx[:, 1] = -15*torch.sin(x[:, 0])

#         return dx

#     def forward_u(self, z):
#         x = z[:, :d]
#         u = z[:, d:]
#         # B = torch.tensor(self.get_B(x.detach().numpy().squeeze()), dtype=torch.float)
#         B = self.B_net(self.transform(x)).view(d, m)
#         return (B @ u.T).T
#         # return B@u

#     def get_B(self, x):
#         zeta = self.transform(torch.tensor(x, dtype=torch.float).unsqueeze(0))
#         return self.B_net(zeta).view(d, m).detach().numpy()
#         return np.array([0, 1]).reshape(d, m)

class LinearModel(NeuralModel): 

    def __init__(self):
        super().__init__()
        self.a_net = nn.Sequential(
            nn.Linear(3, 2, bias=False),
        )
        self.lr = 0.01
