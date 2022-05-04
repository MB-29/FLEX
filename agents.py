import numpy as np
import torch 

from utils import lstsq_update

class Agent:
    
    def __init__(self, x0, m, dynamics, f_theta, theta0, gamma):

        self.x = x0
        self.d = x0.shape[0]
        self.q = theta0.shape[0]
        self.m = m
        
        self.M = np.zeros((self.d, self.d+m, self.d+m))
        for j in range(self.d):
            self.M[j] = 1e-6*np.diag(np.random.randn(self.d+m))
        self.fisher = 1e-6*np.diag(np.random.randn(self.q))

        self.gamma = gamma

        self.f_theta = f_theta
        self.theta = torch.tensor(theta0, requires_grad=True)
        # self.C = theta0.reshape((self.d, -1))
        
        self.dynamics = dynamics
    
    def identify(self, T):

        self.theta_values = np.zeros((T+1, self.q))
        self.theta_values[0] = self.theta.detach().clone().numpy()
        # self.C_values = np.zeros((T+1, self.d, self.d+self.m))
        # self.C_values[0] = self.C
        self.u_values = np.zeros((T, self.m))

        self.optimizer = torch.optim.Adam([self.theta], lr=0.001)

        # self.x_values = torch.zeros(T+1, self.d)
        for t in range(T):

            u_t = self.choose_control(t)

            x_t_ = self.dynamics(self.x, u_t)
            # print(f'x = {self.x}')
            # print(f'u = {u_t}')
            # print(f'x_ = {x_t_}')

            eta = 1/((t+1))
            self.gradient_step(self.x, x_t_, u_t, eta)
            # self.linearized_step(self.x, x_t_, u_t, eta)
            # self.online_OLS(self.x, x_t_, u_t) 

            self.x = x_t_
            # self.x_values[t+1] = torch.tensor(x_t_)

            self.theta_values[t+1] = self.theta.detach().clone().numpy()
            # self.C_values[t+1] = self.C.copy()
            self.u_values[t] = u_t

            # print(self.theta)


        return self.theta_values
    
    def choose_control(self, t):
        u = np.random.randn(self.m)
        # u *= self.gamma / np.linalg.norm(u)
        return u
    def draw_random_control(self, t):
        u = np.random.randn(self.m)
        # u *= self.gamma / np.linalg.norm(u)
        return u
    
    def online_OLS(self, x_t, y_t, u_t):
        z_t = np.zeros(self.d + self.m)
        z_t[:self.d] = x_t
        z_t[self.d:] = u_t
        # print(f'z_t = {z_t}')

        for j in range(self.d):
            prior_gram = self.M[j]
            prior_estimate = self.C[j]
            posterior_estimate, posterior_gram = lstsq_update(
                prior_estimate, prior_gram, z_t, y_t[j])

            self.M[j] = posterior_gram
            # self.C[j] = posterior_estimate

    def gradient_step(self, x, x_, u, eta):

        prediction = self.f_theta(
            torch.tensor(x),
            torch.tensor(u),
            self.theta
            )
        y = torch.tensor(x_)
        loss = (prediction-y)@(prediction-y)
        # print(f'u = {u}')
        # print(f'y = {y}')
        # print(f'loss = {loss}')
        # print(f'prediction = {prediction}')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(f'grad {self.theta.grad}')
        # print(f'eta {eta}')
        
        # d_theta = d_f @ (prediction - x_)
        # with torch.no_grad():
        # self.theta.data -= 0.1*(eta) * self.theta.grad.data
        # self.theta.grad.zero_()
        # print(f'theta {self.theta}')
    def linearized_step(self, x, x_, u, eta):

        sensitivity = np.zeros((self.d, self.q))
        for i in range(self.d):
            prediction = self.f_theta(
                torch.tensor(x),
                torch.tensor(u),
                self.theta
                )
            gradient = torch.autograd.grad(
                prediction[i],
                inputs=self.theta,
            )[0]
            sensitivity[i] = gradient
        print(f'fisher {self.fisher}')
        print(f'sensitivity {sensitivity}')
        print(f'theta {self.theta}')

        self.fisher += sensitivity.T@sensitivity

        self.theta.data += np.linalg.solve(self.fisher, sensitivity.T@(x_-prediction.detach().numpy()))
        return 

