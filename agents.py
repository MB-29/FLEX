import numpy as np
import torch
import torch.nn as nn

from utils import jacobian

class Agent:

    def __init__(self, x0, m, dynamics, model, gamma, dt):

        self.x = x0
        self.d = x0.shape[0]
        self.m = m

        self.model = model
        self.q = sum(parameter.numel() for parameter in model.parameters())
        self.M = 1e-3*np.diag(np.random.rand(self.q))
        self.Mx = 1e-3*np.diag(np.random.rand(self.d))
        self.My = 1e-3*np.diag(np.random.rand(self.d))

        self.gamma = gamma

        self.dt = dt
        self.dynamics = dynamics

    def identify(self, T, T_random=0, test_function=None, plot=False):

        self.T_random =  T_random

        self.u_values = np.zeros((T, self.m))
        self.x_values = np.zeros((T, self.d))
        lr = getattr(self.model, 'lr', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        test_values = np.zeros(T)

        for t in range(T):
            # print(f't = {t}')

            u_t = self.choose_control(t)
            x_dot = self.dynamics(self.x, u_t)

            self.learning_step(self.x, x_dot, u_t)

            self.x += self.dt*x_dot

            self.u_values[t] = u_t
            self.x_values[t] = self.x.copy()

            if test_function is not None:
                with torch.no_grad():
                    test_error = test_function(
                        self.model, self.x, u_t, plot, t=t)
                test_values[t] = test_error.data

            # z = np.array([self.x[0], np.sin(self.x[0])])
            z = torch.zeros(1, self.d + self.m)
            z[:, :self.d] = torch.tensor(self.x)
            z[:, self.d:] = torch.tensor(u_t)
            J = jacobian(self.model, z).detach().numpy()
            # self.M += J[:, None]@J[None, :]
            self.M += J.T @ J
            self.Mx += self.x[:, None]@self.x[None, :]
            # self.y = self.x.copy()
            # self.y[0] = np.sin(self.x[0])
            # self.My += self.y[:, None]@self.y[None, :]
            # print(f'M = {self.M}')
            # print(f'Mx={self.Mx}')
            # print(f'My={self.My}')
            # self.update_information(self.x.copy())

        return test_values

    def learning_step(self, x, x_dot, u):
        z = torch.zeros(1, self.d + self.m)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        u = torch.tensor(u).unsqueeze(0)
        z[:, :self.d] = x
        z[:, self.d:] = u
        prediction = self.model(z)
        x_dot = torch.tensor(x_dot, dtype=torch.float32)
        loss = nn.MSELoss()(prediction.squeeze(), x_dot.squeeze())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # def update_information(self, x):

    def draw_random_control(self, t):
        u = np.random.randn(self.m)
        u *= self.gamma / np.linalg.norm(u)
        return u

    def choose_control(self, t):
        raise NotImplementedError


class Random(Agent):

    def choose_control(self, t):
        u = self.draw_random_control(t)
        return u


class Passive(Agent):

    def choose_control(self, t):
        return np.zeros(self.m)
