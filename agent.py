import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from computations import jacobian, lstsq_update


class Agent:

    def __init__(self, model, d, m, gamma, batch_size=100, dt=None, regularization=1e-7):

        self.d = d
        self.m = m
        self.dt = dt

        self.model = model
        self.q = sum(parameter.numel() for parameter in model.parameters())
        diag = regularization*np.random.rand(self.q)
        self.M = np.diag(diag)
        self.M_inv =np.diag(1/diag)

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

        if self.optimizer == 'OLS':
            J = jacobian(self.model, z, retain_graph=True).detach().numpy()
            prediction = self.model(z)
            theta = parameters_to_vector(self.model.parameters()).detach().numpy()
            c = prediction.detach().numpy().squeeze() - J@theta
            observation = dx_dt - c
            for j in range(self.d):
                v = J[j]
                theta, self.M = lstsq_update(theta, self.M, v, observation[j])
                matrix = self.M_inv@v[:, None]@v[None, :]@self.M_inv
                scalar = -1/(1+v.T@self.M_inv@v)
                increment = scalar*matrix
                self.M_inv += increment
            vector_to_parameters(torch.tensor(
                theta, dtype=torch.float), self.model.parameters())
            return
            
        dx_dt = torch.tensor(dx_dt, dtype=torch.float, requires_grad=False)
        self.target_values  = torch.cat((self.target_values[1:, :], dx_dt.unsqueeze(0)), dim=0)
        self.z_values  = torch.cat((self.z_values[1:, :], z), dim=0)
        predictions = self.model(self.z_values)

        loss = self.loss_function(predictions, self.target_values)
        self.optimizer.zero_grad() , loss.backward() ; self.optimizer.step()

        if self.__class__.__name__ != 'Flex':
            return

        J = jacobian(self.model, z, retain_graph=True).detach().numpy()
        for j in range(self.d):
            v = J[j][:, None]
            matrix = self.M_inv@v@v.T@self.M_inv
            scalar = -1/(1+v.T@self.M_inv@v)
            increment = scalar*matrix
            self.M_inv += increment
        self.M += J.T @ J 

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

