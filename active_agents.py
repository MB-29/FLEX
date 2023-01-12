import numpy as np
import torch

from agents import Agent
from computations import jacobian, compute_gradient, maximizer_quadratic

class Active(Agent):

    def predict_x(self, x, u):

        for param in self.model.parameters():
            param.requires_grad = False

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x
        z[:, self.d:] = torch.tensor(u, dtype=torch.float)
        x_dot = self.model(z)
        x_ = x + self.dt * x_dot

        for param in self.model.parameters():
            param.requires_grad = True
        return x_

    def predict_x_(self, u):
        for param in self.model.parameters():
            param.requires_grad = False

        x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x
        z[:, self.d:] += u
        dx = self.model(z)
        x_ = x + self.dt * dx

        z_ = torch.zeros(1, self.d + self.m)
        z_[:, :self.d] += x_
        z_[:, self.d:] += u
        dx_ = self.model(z_)
        x__ = x_ + self.dt * dx_

        for param in self.model.parameters():
            param.requires_grad = True

        return x__


class Gradient(Active):

    def maximum_utility(self, x, t, n_gradient=10):
        u = torch.randn(1, self.m)
        u *= self.gamma / torch.linalg.norm(u)
        if self.m == 1:
            utility = self.utility(u, x, t)
            utility_ = self.utility(-u, x, t)
            # print(f'u: {u}, -u: {-u}')
            # print(f'u: {utility}, -u: {utility_}')
            u = -u if utility_ > utility else u
            # print(f'u = {u}')
            return u.view(1).detach().numpy()

        u.requires_grad = True
        designer = torch.optim.Adam([u], lr=0.1)
        # print(f't = {t}')
        for step_index in range(n_gradient):
            loss = -self.utility(u, x, t)
            # print(f'loss = {loss}')
            designer.zero_grad()
            loss.backward()
            designer.step()

            u.data *= self.gamma/torch.linalg.norm(u.data)
        u = u.squeeze().detach().numpy()
        return u

    def policy(self, x, t):
        u = self.maximum_utility(x, t)
        # print(f't={t}, u = {u}')
        return u


class GradientDesign(Gradient):

    def utility_(self, u, t):
        # z.requires_grad = True
        u = torch.tensor(u).unsqueeze(0)
        # print(z)
        x_ = self.predict_x(u)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x_
        M_z = torch.tensor(self.M, dtype=torch.float)
        J = jacobian(self.model, z)[1]
        M_z += J@J.T
        M_inv = torch.linalg.inv(M_z)
        y = jacobian(self.model, torch.randn(1, self.d+self.m))
        uncertainty = 0
        uncertainty += y[1]@M_inv@y[1]
        # print(f'u = {u}, uncertainty = {uncertainty}')

        return uncertainty

    def utility(self, u, x, t):
        # z.requires_grad = True
        # print(z)
        # with torch.no_grad():
        x_ = self.predict_x(x, u)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x_
        M_z = torch.tensor(self.M, dtype=torch.float)
        if self.m == 1:
            J = jacobian(self.model, z)[1]
            M_z += J[:, None]@J[None, :]
            # J = jacobian(self.model, z)
            # M_z += J.T@J
            uncertainty = (1/self.q)*torch.logdet(M_z)
            return uncertainty

        # with torch.no_grad():
        # print(f'y = {y}')
        # J = torch.autograd.grad((y**2).sum(), [self.model.direction_net[0].weight], retain_graph=True)[0]
        # print(f'J = {J}, requires={J.requires_grad}')
        # print(f'M_z = {M_z}')
        y = self.model(z)
        tensor_gradients = torch.autograd.grad(
            y[:, 1],
            self.model.parameters(),
            create_graph=True
        )
        derivatives = []
        for tensor in tensor_gradients:
            derivatives.append(tensor.view(-1, 1))
        J = torch.cat(derivatives).squeeze()
        M_inv = torch.inverse(M_z)
        # M_z += J.T@J
        uncertainty = J.T @ (M_inv@J)

        # g = torch.autograd.grad((J**2).sum(), u)
        # print(f'g = {g}')
        # print(f'uncertainty {uncertainty}')
        # print(f'u = {u}, uncertainty = {uncertainty}')

        return uncertainty


class Variation(Active):
    def utility(self, u, x, t):
        # z.requires_grad = True
        # print(z)
        x_ = self.predict_x(u)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x_
        uncertainty = 0
        J = jacobian(self.model, z)
        uncertainty = torch.sum(J**2)
        return uncertainty


class Spacing(Gradient):

    def utility(self, u, x, t):

        x_ = self.predict_x(x, u)
        z_ = torch.zeros(1, self.d + self.m)
        z_[:, :self.d] += x_
        # z_[:, self.d:] = u
        # z_values = torch.zeros(t, self.d + self.m)
        # z_values[:, :self.d] = torch.tensor(self.x_values[:t])
        z_values = torch.tensor(self.z_values, dtype=torch.float)
        # z_values[:, self.d:] = torch.tensor(self.u_values[:t])

        past = z_values
        future = z_
        # past = self.model.transform(z_values)
        # future = self.model.transform(z_)


        differences = past - future
        # print(f'x {x}, u = {u}')
        # print(f'transform {self.model.transform(z)}')
        # print(f'future {future}')
        distance = torch.mean(differences**2)
        # print(f'distance {distance}')
        return distance


class D_optimal(Active):

    def policy(self, x, t):
        # epsilon = np.random.random()
        # if epsilon < 0.3:
        #     return self.draw_random_control(t)

        z = torch.zeros(1, self.d + self.m)
        x = torch.tensor(x, dtype=torch.float, requires_grad=True)
        # u = torch.tensor(self.u, dtype=torch.float)
        u = torch.zeros(self.m, requires_grad=True)
        z[:, :self.d] = x
        z[:, self.d:] = u

        j = np.random.choice(self.d)
        # j = 1

        y = self.model(z)
        df_dtheta = compute_gradient(self.model, y[:, j])

        D = np.zeros((self.q, self.d))
        y = self.model(z)
        df_dx = torch.autograd.grad(y[:, j], x, create_graph=True)[
            0].unsqueeze(0)

        for i in range(self.d):
            d2a_dxidtheta = compute_gradient(
                self.model, df_dx[:, i], retain_graph=True, allow_unused=True)
            D[:, i] = d2a_dxidtheta
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
        # return u.detach().numpy()
        u = maximizer_quadratic(self.M_inv, B, v, self.gamma)
        u *= self.gamma / np.linalg.norm(u)
        return u

