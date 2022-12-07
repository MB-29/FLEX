import numpy as np
import torch
import torch.nn as nn

# from active_agents import D_optimal, Spacing
from computations import jacobian

class Agent:

    def __init__(self, model, d, m, gamma):

        self.d = d
        self.m = m

        self.model = model
        self.q = sum(parameter.numel() for parameter in model.parameters())
        self.M = 1e-3*np.diag(np.random.rand(self.q))
        # self.Mx = 1e-3*np.diag(np.random.rand(self.d))
        # self.My = 1e-3*np.diag(np.random.rand(self.d))

        self.gamma = gamma

        self.lr = getattr(self.model, 'lr', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()



    def learning_step(self, x, x_dot, u):
        z = torch.zeros(1, self.d + self.m, requires_grad=False)
        x = torch.tensor(x, dtype=torch.float, requires_grad=False).unsqueeze(0)
        u = torch.tensor(u, dtype=torch.float, requires_grad=False).unsqueeze(0)
        z[:, :self.d] = x
        z[:, self.d:] = u
        x_dot = torch.tensor(x_dot, dtype=torch.float, requires_grad=False)

        prediction = self.model(z)
        loss = self.loss_function(prediction.squeeze(), x_dot.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 


        # zeta = self.model.transform(x)

        # 08/23/2022 : zeta instead of z
        # 09/02/2022 : z instead of zeta
        J = jacobian(self.model, z).detach().numpy()
        # self.M += J[:, None]@J[None, :]
        self.M += J.T @ J
        # self.Mx += self.x[:, None]@self.x[None, :]

    def draw_random_control(self, t):
        u = np.random.randn(self.m)
        u *= self.gamma / np.linalg.norm(u)
        return u

    def policy(self, x, t):
        raise NotImplementedError


class Random(Agent):

    def policy(self, x, t):
        u = self.draw_random_control(t)
        return u


class Passive(Agent):

    def policy(self, x, t):
        return np.zeros(self.m)
