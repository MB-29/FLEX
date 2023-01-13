import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector

# from active_agents import D_optimal, Spacing
from computations import jacobian, lstsq_update


class Agent:

    def __init__(self, model, d, m, gamma, batch_size=100, dt=None):

        self.d = d
        self.m = m
        self.dt = dt

        self.model = model
        self.q = sum(parameter.numel() for parameter in model.parameters())
        diag = 1e-3*np.random.rand(self.q)
        self.M = np.diag(diag)
        self.M_inv =np.diag(1/diag)
        # self.Mx = 1e-3*np.diag(np.random.rand(self.d))
        # self.My = 1e-3*np.diag(np.random.rand(self.d))

        self.gamma = gamma

        self.batch_size = batch_size
        self.z_values = torch.zeros(batch_size, d+m)
        self.target_values = torch.zeros(batch_size, d)
        self.lr = getattr(self.model, 'lr', None)
        self.optimizer = 'OLS' if self.lr is None else torch.optim.Adam(self.model.parameters(), lr=self.lr)
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

        if self.optimizer == 'OLS':
            J = jacobian(self.model, z).detach().numpy()
            prediction = self.model(z)
            theta = parameters_to_vector(self.model.parameters()).detach().numpy()
            c = prediction.detach().numpy().squeeze() - J@theta
            observation = dx_dt - c
            # print(c.shape)
            for j in range(self.d):
                # print(f'j={j}, M={self.M}')
                v = J[j]
                theta, self.M = lstsq_update(theta, self.M, v, observation[j])
                matrix = self.M_inv@v[:, None]@v[None, :]@self.M_inv
                scalar = -1/(1+v.T@self.M_inv@v)
                increment = scalar*matrix
                # print(f'scalar {scalar}')
                # print(f'matrix {matrix}')
                self.M_inv += increment
            vector_to_parameters(torch.tensor(
                theta, dtype=torch.float), self.model.parameters())
            return
            
        dx_dt = torch.tensor(dx_dt, dtype=torch.float, requires_grad=False)
        self.target_values  = torch.cat((self.target_values[1:, :], dx_dt.unsqueeze(0)), dim=0)
        self.z_values  = torch.cat((self.z_values[1:, :], z), dim=0)
        # print(f'z_values = {self.z_values}')
        # print(f'target_values = {self.target_values}')
        # print(f'learn')
        predictions = self.model(self.z_values)
        # print(f'residual {predictions-self.target_values}')

        loss = self.loss_function(predictions, self.target_values)
        # print(f'learning loss {loss}')
        self.optimizer.zero_grad() , loss.backward() ; self.optimizer.step()
        # zeta = self.model.transform(x)

        # 08/23/2022 : zeta instead of z
        # 09/02/2022 : z instead of zeta
        # self.M += J[:, None]@J[None, :]
        # print('update')
        J = jacobian(self.model, z).detach().numpy()
        # self.M += J.T @ J 
        for j in range(self.d):
            # print(f'matrix {self.M_inv}')
            v = J[j][:, None]
            # print(f'v {v}')
            matrix = self.M_inv@v@v.T@self.M_inv
            scalar = -1/(1+v.T@self.M_inv@v)
            increment = scalar*matrix
            # print(f'scalar {scalar}')
            # print(f'matrix {matrix}')
            self.M_inv += increment
        # diff = np.abs(self.M_inv - np.linalg.inv(self.M))/self.M_inv
        # self.J = J

        # print(f'diff = {diff}')
        # self.Mx += self.x[:, None]@self.x[None, :]

    def draw_random_control_max(self):
        u = np.random.randn(self.m)
        u *= self.gamma / np.linalg.norm(u)
        return u
    def draw_random_control(self):
        u = 2*np.random.rand(self.m)-1
        u *= self.gamma / np.sqrt(self.m)
        return u

    def policy(self, x, t):
        raise NotImplementedError

class Random(Agent):

    def policy(self, x, t):
        u = self.draw_random_control()
        return u

class MaxRandom(Agent):

    def policy(self, x, t):
        u = self.draw_random_control_max()
        return u


class Passive(Agent):

    def policy(self, x, t):
        return np.zeros(self.m)
