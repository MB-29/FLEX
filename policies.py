import numpy as np
import torch

from agents import Agent
from computations import jacobian, compute_gradient, maximizer_quadratic



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

class Uniform(Agent):

    def maximum_utility(self, x, t, n_gradient=10):
        u = torch.randn(1, self.m)
        u *= self.gamma / torch.linalg.norm(u)
        if self.m == 1:
            utility = self.utility(u, x, t)
            utility_ = self.utility(-u, x, t)
            u = -u if utility_ > utility else u
            return u.view(1).detach().numpy()

        u.requires_grad = True
        designer = torch.optim.Adam([u], lr=0.1)
        for step_index in range(n_gradient):
            loss = -self.utility(u, x, t)
            designer.zero_grad()
            loss.backward()
            designer.step()

            u.data *= self.gamma/torch.linalg.norm(u.data)
        u = u.squeeze().detach().numpy()
        return u

    def policy(self, x, t):
        u = self.maximum_utility(x, t)
        return u

    def predict_x(self, x, u):

        for param in self.model.parameters():
            param.requires_grad = False

        x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        z = torch.cat((x, u), dim=1)
        x_dot = self.model(z)
        x_ = x + self.dt * x_dot

        for param in self.model.parameters():
            param.requires_grad = True
        return x_

    def utility(self, u, x, t):

        x_ = self.predict_x(x, u)
        z_ = torch.zeros(1, self.d + self.m)
        z_[:, :self.d] += x_
        z_values = self.z_values.clone()

        past = z_values
        future = z_

        differences = past - future
        distance = torch.mean(differences**2)
        return distance


class Flex(Agent):

    def policy(self, x, t):

        z = torch.zeros(1, self.d + self.m)
        x = torch.tensor(x, dtype=torch.float, requires_grad=True)
        u = torch.zeros(self.m, requires_grad=True)
        z[:, :self.d] = x
        z[:, self.d:] = u

        j = np.random.choice(self.d)

        y = self.model(z)
        df_dtheta = compute_gradient(self.model, y[:, j])

        D = np.zeros((self.q, self.d))
        y = self.model(z)
        df_dx = torch.autograd.grad(y[:, j], x, create_graph=True)[
            0].unsqueeze(0)

        for i in range(self.d):
            d2f_dxidtheta = compute_gradient(
                self.model, df_dx[:, i], retain_graph=True, allow_unused=True)
            D[:, i] = d2f_dxidtheta
        try:
            B_ = self.dt * self.model.get_B(x)
        except AttributeError:
            B_ = np.zeros((self.d, self.m))
            y = self.model(z)
            for i in range(self.d):
                dfi_du = torch.autograd.grad(y[:, i], u, retain_graph=True)
                B_[i, :] = dfi_du[0].numpy()
        B = D @ B_
        v = df_dtheta.detach().numpy()
        u = maximizer_quadratic(self.M_inv, B, v, self.gamma)
        u *= self.gamma / np.linalg.norm(u)
        return u

class Episodic(Agent):

    def __init__(self, model, d, m, gamma, planning_horizon, n_gradient, **kwargs):
        self.planning_horizon = planning_horizon
        self.n_gradient = n_gradient
        super().__init__(model, d, m, gamma, **kwargs)

        U = np.random.randn(planning_horizon, m) 
        self.planned_inputs = self.gamma * U / np.linalg.norm(U, axis=1)[:, None]

    def policy(self, x, t):
        schedule = t % self.planning_horizon
        if schedule == 0:
            self.plan_inputs(x)
        return self.planned_inputs[schedule]

    def plan_inputs(self, x):
        print(f'planning')
        U = torch.randn(self.planning_horizon, self.m)
        U *= self.gamma / torch.linalg.norm(U, dim=1).unsqueeze(1)
        U.requires_grad = True
        designer = torch.optim.Adam([U], lr=10.)
        for step_index in range(self.n_gradient):
            loss = -self.information_gain(x, U)
            
            designer.zero_grad() ; loss.backward() ; designer.step()
        U.data *= self.gamma/torch.linalg.norm(U.data, dim=1).unsqueeze(1)
        self.planned_inputs = U.squeeze().detach().numpy()

        for param in self.model.parameters():
            param.requires_grad = True


    def information_gain(self, x, U):
        x = torch.tensor(x, requires_grad=True).unsqueeze(0)
        M = torch.tensor(self.M, requires_grad=True)
        
        for t in range(self.planning_horizon):
            
            u = U[t].unsqueeze(0)
            z = torch.cat((x, u), dim=1)
            x_dot = self.model(z)
            x = x + self.dt * x_dot
            u_ = torch.zeros(1, self.m)
            if t < self.planning_horizon-1:
                u_ = U[t+1].unsqueeze(0)
            z_ = torch.cat((x, u_), dim=1)

            for param in self.model.parameters():
                param.requires_grad = True
            V = jacobian(self.model, z_, create_graph=True, retain_graph=True)
            for param in self.model.parameters():
                param.requires_grad = False

            M = M + V.T @ V 

        gain = torch.det(M)/torch.det(torch.tensor(self.M))

        return gain