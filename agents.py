import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector

# from active_agents import D_optimal, Spacing
from computations import jacobian, lstsq_update


class Agent:

    def __init__(self, model, d, m, gamma, sigma=1e-2, batch_size=48):

        self.d = d
        self.m = m

        self.model = model
        self.q = sum(parameter.numel() for parameter in model.parameters())
        self.M = 1e-3*np.diag(np.random.rand(self.q))
        # self.Mx = 1e-3*np.diag(np.random.rand(self.d))
        # self.My = 1e-3*np.diag(np.random.rand(self.d))

        self.gamma = gamma
        self.sigma = sigma

        self.z_values = torch.zeros(batch_size, d+m)
        self.target_values = torch.zeros(batch_size, d)
        self.lr = getattr(self.model, 'lr', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()

    def learning_step(self, x, dx_dt, u):
        z = torch.zeros(1, self.d + self.m, requires_grad=False)
        x = torch.tensor(x, dtype=torch.float,
                         requires_grad=False).unsqueeze(0)
        u = torch.tensor(u, dtype=torch.float,
                         requires_grad=False).unsqueeze(0)
        z[:, :self.d] = x
        z[:, self.d:] = u

        # print(z)


        J = jacobian(self.model, z).detach().numpy()

        if getattr(self.model, 'linear', False):
            prediction = self.model(z)
            theta = parameters_to_vector(self.model.parameters()).detach().numpy()
            c = prediction.detach().numpy().squeeze() - J@theta
            observation = dx_dt - c
            # print(c.shape)
            for j in range(self.d):
                # print(f'j={j}, M={self.M}')
                theta, self.M = lstsq_update(theta, self.M, J[j], observation[j], self.sigma)
            vector_to_parameters(torch.tensor(
                theta, dtype=torch.float), self.model.parameters())
            return
            
        dx_dt = torch.tensor(dx_dt, dtype=torch.float, requires_grad=False)
        self.target_values  = torch.cat((self.target_values[1:, :], dx_dt.unsqueeze(0)), dim=0)
        self.z_values  = torch.cat((self.z_values[1:, :], z), dim=0)
        # print(f'z_values = {self.z_values}')
        # print(f'target_values = {self.target_values}')

        predictions = self.model(self.z_values)
        loss = self.loss_function(predictions, self.target_values)
        self.optimizer.zero_grad() , loss.backward() ; self.optimizer.step()
        # zeta = self.model.transform(x)

        # 08/23/2022 : zeta instead of z
        # 09/02/2022 : z instead of zeta
        # self.M += J[:, None]@J[None, :]
        self.M += J.T @ J / self.sigma**2
        # self.J = J

        # print(f'J = {J}')
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
