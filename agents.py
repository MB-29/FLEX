import numpy as np
import torch
import torch.nn as nn

from computations import jacobian

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

        self.u = np.zeros(self.m)
        self.u_values = np.zeros((T, self.m))
        self.x_values = np.zeros((T, self.d))
        # self.lr_a = getattr(self.model, 'lr', 0.001)
        # self.lr_b = getattr(self.model, 'lr', 0.001)
        # self.optimizer_a = torch.optim.Adam(self.model.parameters(), lr=self.lr_a)
        # self.optimizer_b = torch.optim.Adam(self.model.B_net.parameters(), lr=self.lr_b)
        self.lr = getattr(self.model, 'lr', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        test_values = np.zeros(T)

        for t in range(T):
            # print(f't = {t}')

            u_t = self.choose_control(t)
            self.u = u_t
            x_dot = self.dynamics(self.x, u_t.copy())

            self.learning_step(self.x, x_dot, u_t)

            self.x += self.dt*x_dot

            self.u_values[t] = u_t.copy()
            self.x_values[t] = self.x.copy()

            if test_function is not None:
                with torch.no_grad():
                    test_error = test_function(
                        self.model, self.x, u_t, plot, t=t)
                test_values[t] = test_error.data

            z = torch.zeros(1, self.d + self.m)
            z[:, :self.d] = torch.tensor(self.x)
            z[:, self.d:] = torch.tensor(u_t)
            x = torch.tensor(self.x, dtype=torch.float).unsqueeze(0)
            # zeta = self.model.transform(x)

            # 08/23/2022 : zeta instead of z
            # 09/02/2022 : z instead of zeta
            J = jacobian(self.model, z).detach().numpy()
            # self.M += J[:, None]@J[None, :]
            self.M += J.T @ J
            self.Mx += self.x[:, None]@self.x[None, :]

        return test_values

    def learning_step(self, x, x_dot, u):
        z = torch.zeros(1, self.d + self.m, requires_grad=False)
        x = torch.tensor(x, dtype=torch.float, requires_grad=False).unsqueeze(0)
        u = torch.tensor(u, dtype=torch.float, requires_grad=False).unsqueeze(0)
        z[:, :self.d] = x
        z[:, self.d:] = u
        # print(f'z {z}')

        x_dot = torch.tensor(x_dot, dtype=torch.float, requires_grad=False)
        prediction = self.model.forward(z)
        # print(f'prediction {prediction}')
        
        # x_dot_ = np.array([[0.0], [3.0]]) @ u
        # x_dot = torch.tensor(x_dot_, dtype=torch.float, requires_grad=False)
        # prediction = self.model.forward_u(z)
        # print(f'x_dot {x_dot}')
        loss = nn.MSELoss()(prediction.squeeze(), x_dot.squeeze())
        
        self.optimizer.zero_grad()
        # print(f'before {self.model.B_net[0].weight.grad}')
        loss.backward()
        # print(f'after {self.model.B_net[0].weight.grad}')
        self.optimizer.step() 

        # self.optimizer_a.zero_grad() ; self.optimizer_b.zero_grad()
        # loss.backward()
        # self.optimizer_a.step() ; self.optimizer_b.step() 

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
