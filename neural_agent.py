import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def jacobian(model, z):
    y = model(z)
    _, d = y.shape 
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
    return J

class Agent:

    def __init__(self, x0, m, dynamics, model, gamma, dt):

        self.x = x0
        self.d = x0.shape[0]
        self.m = m

        self.model = model
        self.q = sum(parameter.numel() for parameter in model.parameters())
        self.M = 1e-6*np.diag(np.random.rand(self.d))

        self.gamma = gamma

        self.dt = dt
        self.dynamics = dynamics

    def identify(self, T, test_function=None, plot=False):

        self.u_values = np.zeros((T, self.m))
        self.x_values = np.zeros((T, self.d))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        test_values = np.zeros(T)

        for t in range(T):

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
            # self.M += z[:, None]@z[None, :]
            # self.update_information(self.x.copy())

        return test_values

    def learning_step(self, x, x_dot, u):
        z = torch.zeros(1, self.d + self.m)
        x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
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

    def optimal_design_(self, t, n_gradient=2):
        # print(f't = {t}')
        u = self.gamma * torch.randn(1, self.m)
        # u.requires_grad = True
        # designer = torch.optim.Adam([u], lr=0.0001)
        # for step_index in range(n_gradient):
        #     # print(loss)

        #     designer.zero_grad() ; loss.backward() ; designer.step()
        #     with torch.no_grad():
        #         u *= self.gamma / torch.linalg.norm(u)
        return u.detach().numpy()

    def gradient_design(self, t, n_gradient=20):
        u = self.gamma * torch.randn(self.m)
        u.requires_grad = True

        designer = torch.optim.Adam([u], lr=0.01)
        i = np.random.randint(0, self.d)
        for step_index in range(n_gradient):
            z = torch.zeros(self.d + self.m)
            z[:self.d] = torch.tensor(self.x)
            z[self.d:] = u
            prediction = self.model(z)[i]
            tensor_gradients = torch.autograd.grad(
                prediction,
                self.model.parameters(),
                create_graph=True)
            derivatives = []
            for tensor in tensor_gradients:
                derivatives.append(tensor.view(-1, 1))
            gradient = torch.cat(derivatives).squeeze()
            Q = torch.tensor(np.linalg.inv(self.M), dtype=torch.float32)
            loss = - gradient.T@Q@gradient
            designer.zero_grad()
            loss.backward()
            designer.step()

            # print(f'step {step_index}, loss={loss:2e}')

            u.data *= self.gamma / np.linalg.norm(u.data)

        increment = (gradient[:, None]@gradient[None, :]).detach().numpy()
        # print(f'increment {increment}')
        self.M += increment

        return u.detach().numpy()


class Random(Agent):

    def choose_control(self, t):
        return self.draw_random_control(t)


class Passive(Agent):

    def choose_control(self, t):
        return np.zeros(self.m)


class Active(Agent):

    def predict_x(self, x, u):
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x
        z[:, self.d:] = u
        dx = self.model(z)
        # dx = self.model.forward_u(dx, u)
        x_ = x + self.dt * dx
        return x_

    def maximum_utility(self, t):
        u = np.random.randn(self.m)
        u *= self.gamma / np.linalg.norm(u)
        utility = self.utility(u, t)
        utility_ = self.utility(-u, t)
        # print(f'u: {u}, -u: {-u}')
        # print(f'u: {utility}, -u: {utility_}')
        u = -u if utility_ > utility else u
        # print(u)  
        return u

    def choose_control(self, t):
        if t < 50 :
        # or t%100 == 0:
            return self.draw_random_control(t)
        u = self.maximum_utility(t)
        # print(f't={t}, u = {u}')
        return u


class OptimalDesign(Active):

    def utility__(self, u, t):
        # z.requires_grad = True
        u = torch.tensor(u).unsqueeze(0)
        x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        # print(z)
        with torch.no_grad():
            x_ = self.predict_x(x, u)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x_
        y = self.model(z)
        uncertainty = 0
        for i in range(self.d):
            tensor_gradients = torch.autograd.grad(
                y[:, i],
                self.model.net.parameters(),
                # create_graph=True,
                retain_graph=True
            )
            for tensor in tensor_gradients:
                uncertainty += torch.sum(tensor**2)
        return uncertainty
    def utility(self, u, t):
        # z.requires_grad = True
        u = torch.tensor(u).unsqueeze(0)
        x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        # print(z)
        with torch.no_grad():
            x_ = self.predict_x(x, u)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x_
        uncertainty = 0
        J = jacobian(self.model, z)
        self.J = J
        uncertainty = torch.sum(J**2)
        return uncertainty

    def utility_(self, u, t):
        # z.requires_grad = True
        u = torch.tensor(u).unsqueeze(0)
        x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        # print(z)
        with torch.no_grad():
            x_ = self.predict_x(x, u)
        z = torch.zeros(1, self.d + self.m)
        z[:, :self.d] = x_
        y = self.model(z)
        uncertainty = 0
        M_x = torch.tensor(self.M, requires_grad=True)
        for i in range(self.d):
            output = y[:, i]
            gradient = torch.autograd.grad(
                output,
                self.model.net.parameters(),
                # create_graph=True,
                retain_graph=True
            )
            increment =  (gradient[:, None]@gradient[None, :])
            M_x += increment
        uncertainty = torch.logdet(M_x)

        return uncertainty



class Spacing(Active):

    def utility(self, u, t):

        x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        u = torch.tensor(u).unsqueeze(0)
        x_ = self.predict_x(x, u)
        z_ = torch.zeros(1, self.d + self.m)
        z_[:, :self.d] = x_
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


class Periodic(Agent):

    def choose_control(self, t):
        if t < 50:
            return self.draw_random_control(t)
        return self.gamma * np.sign(np.sin(2*np.pi*t/100))


# class Oracle(Agent):

#     def choose_control(self, t):
#         if np.allclose(self.M, 0) or t < 100:
#             return self.draw_random_control(t)
#         A = np.array([[1, self.dt], [self.dt, 1-0.1*self.dt]])
#         B = np.array([[0], [self.dt]])
#         u = greedy_optimal_input(self.M, A, B, self.x, self.gamma)
#         u *= self.gamma / np.linalg.norm(u)
#         # print(u)
#         return u
