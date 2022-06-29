from venv import create
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import linear_D_optimal

def compute_gradient(model, output, **grad_kwargs):
    # print(grad_kwargs)
    tensor_gradients = torch.autograd.grad(
        output,
        model.parameters(),
        **grad_kwargs
        )
    derivatives = []
    # print(f'output = {output}')
    for index, tensor in enumerate(tensor_gradients):
        if tensor is None:
            tensor = torch.zeros_like(list(model.parameters())[index])
        derivatives.append(tensor.view(-1, 1))
    gradient = torch.cat(derivatives).squeeze()
    return gradient

def jacobian(model, z):
    y = model(z)
    # g = torch.autograd.grad(z.sum(), u)
    # print(f'g = {g}')
    batch_size, d = y.shape
    assert batch_size == 1
    q = sum(parameter.numel() for parameter in model.parameters())
    J = torch.zeros(d, q, dtype=torch.float)
    for i in range(d):
        tensor_gradients = torch.autograd.grad(
            y[:, i],
            model.parameters(),
            create_graph=True,
            retain_graph=True
        )
        derivatives = []
        for tensor in tensor_gradients:
            derivatives.append(tensor.view(-1, 1))
        gradient = torch.cat(derivatives).squeeze()
        J[i] = gradient
        # print(f'J[i] = {J[i]}')
    return J


class Agent:

    def __init__(self, x0, m, dynamics, model, gamma, dt):

        self.x = x0
        self.d = x0.shape[0]
        self.m = m

        self.model = model
        self.q = sum(parameter.numel() for parameter in model.parameters())
        self.M = 1e-3*np.diag(np.random.rand(self.q))
        self.Mx = 1e-3*np.diag(np.random.rand(self.d))

        self.gamma = gamma

        self.dt = dt
        self.dynamics = dynamics

    def identify(self, T, test_function=None, plot=False):

        self.u_values = np.zeros((T, self.m))
        self.x_values = np.zeros((T, self.d))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
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
            self.M += J.T@ J
            self.Mx += self.x[:, None]@self.x[None, :]
            # print(f'M = {self.M}')
            # print(f'Mx={self.Mx}')
            # self.update_information(self.x.copy())

        return test_values

    def learning_step(self, x, x_dot, u):
        z = torch.zeros(1, self.d + self.m)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        u = torch.tensor(u).unsqueeze(0)
        z[:, :self.d] = x
        z[:, self.d:] = u
        # dx = self.model.forward(z)
        # dx = self.model.forward_x(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        # prediction = self.model.forward_u(dx, torch.tensor(u, dtype=torch.float32).unsqueeze(0))
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


class Random(Agent):

    def choose_control(self, t):
        return self.draw_random_control(t)


class Passive(Agent):

    def choose_control(self, t):
        return np.zeros(self.m)


class Active(Agent):

    def predict_x(self, u):


        for param in self.model.parameters():
            param.requires_grad = False

        x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x
        z[:, self.d:] += u
        # with torch.no_grad():
        #     dx = self.model.forward_x(z)
        # dx = self.model.forward_u(dx, u)
        dx = self.model(z)
        x_ = x + self.dt * dx

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
        # dx = self.model.forward_u(dx, u)
        x_ = x + self.dt * dx

        z_ = torch.zeros(1, self.d + self.m)
        z_[:, :self.d] += x_
        z_[:, self.d:] += u
        dx_ = self.model(z_)
        # dx = self.model.forward_u(dx, u)
        x__ = x_ + self.dt * dx_

        for param in self.model.parameters():
            param.requires_grad = True

        return x__

class Gradient(Active):

    def maximum_utility(self, t, n_gradient=10):
        u = torch.randn(1, self.m)
        u *= self.gamma / torch.linalg.norm(u)
        if self.m == 1:
            utility = self.utility(u, t)
            utility_ = self.utility(-u, t)
            # print(f'u: {u}, -u: {-u}')
            # print(f'u: {utility}, -u: {utility_}')
            u = -u if utility_ > utility else u
            return u.squeeze().detach().numpy()

        u.requires_grad=True
        designer = torch.optim.Adam([u], lr=0.1)
        # print(f't = {t}')
        for step_index in range(n_gradient):
            loss = -self.utility(u, t)
            # print(f'loss = {loss}')
            designer.zero_grad()
            loss.backward()
            designer.step()

            u.data *= self.gamma/torch.linalg.norm(u.data)
        # print(u)
        return u.squeeze().detach().numpy()

    def choose_control(self, t):
        if t < 50:
            # or t%100 == 0:
            return self.draw_random_control(t)
        u = self.maximum_utility(t)
        # print(f't={t}, u = {u}')
        return u


class OptimalDesign(Gradient):

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

    def utility(self, u, t):
        # z.requires_grad = True
        # print(z)
        # with torch.no_grad():
        x_ = self.predict_x_(u)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x_
        M_z = torch.tensor(self.M, dtype=torch.float)
        if self.m == 1:
            # J = jacobian(self.model, z)[1]
            # M_z += J[:, None]@J[None, :]
            J = jacobian(self.model, z)
            M_z += J.T@J
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
    def utility(self, u, t):
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

    def utility(self, u, t):

        x_ = self.predict_x(u)
        z_ = torch.zeros(1, self.d + self.m)
        z_[:, :self.d] += x_
        # z_[:, self.d:] = u
        z_values = torch.zeros(t, self.d + self.m)
        z_values[:, :self.d] = torch.tensor(self.x_values[:t])
        # z_values[:, self.d:] = torch.tensor(self.u_values[:t])
        past = self.model.transform(z_values)
        future = self.model.transform(z_)
        differences = past - future
        # print(f'x {x}, u = {u}')
        # print(f'transform {self.model.transform(z)}')
        # print(f'future {future}')
        distance = torch.mean(differences**2)
        # print(f'distance {distance}')
        return distance

class Linearized(Active):

    def choose_control(self, t):
        if t < 50:
            # or t%100 == 0:
            return self.draw_random_control(t)

        z = torch.zeros(1, self.d + self.m)
        x = torch.tensor(self.x, dtype=torch.float, requires_grad=True)
        z[:, :self.d] = x

        y = self.model(z)
        # print(f'y = {y}')
        da_dtheta = compute_gradient(self.model, y[:, 1])
        # print(f'da_dtheta = {da_dtheta}')

        D = np.zeros((self.q, self.d))
        y = self.model(z)
        da_dx = torch.autograd.grad(y[:, 1], x, create_graph=True)[0].unsqueeze(0)
        # print(f'grad = {self.model.net[2].bias.grad}')
        # print(f'da_dx = {da_dx}')
        # g = torch.autograd.grad(da_dx[:, 1], self.model.net[2].bias, allow_unused=True)
        # print(f'g = {g}')

        for i in range(self.d):
            d2a_dxidtheta = compute_gradient(self.model, da_dx[:, i], retain_graph=True, allow_unused=True)
            D[:, i] = d2a_dxidtheta
        # print(f'D = {D}')
        
        B = D @ self.model.get_B(self.x)

        v = da_dtheta.detach().numpy()
        u = linear_D_optimal(self.M, B, v, self.gamma)

        return u




class Periodic(Agent):

    def choose_control(self, t):
        if t < 50:
            return self.draw_random_control(t)
        return self.gamma * np.sign(np.sin(2*np.pi*t/100))


