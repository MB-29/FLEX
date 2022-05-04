import numpy as np
import torch 

from utils import lstsq_update

class Agent:
    
    def __init__(self, x0, m, dynamics, net, gamma):

        self.x = x0
        self.d = x0.shape[0]
        self.m = m

        self.net = net
        
        self.M = np.zeros((self.d, self.d+m, self.d+m))
        for j in range(self.d):
            self.M[j] = 1e-6*np.diag(np.random.randn(self.d+m))
        # self.fisher = 1e-6*np.diag(np.random.randn(self.q))

        self.gamma = gamma

        
        self.dynamics = dynamics
    
    def identify(self, T, test_function=None):

        self.u_values = np.zeros((T, self.m))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        test_values = []

        for t in range(T):

            u_t = self.choose_control(t)

            x_t_ = self.dynamics(self.x, u_t)

            self.gradient_step(self.x, x_t_, u_t)

            self.x = x_t_

            # print(f'u = {u_t}, x = {self.x}')

            self.u_values[t] = u_t

            if test_function is not None:

                test_error = test_function(self.net)
                test_values.append(test_error.data)

        return test_values
    
    def choose_control(self, t):
        u =  np.random.randn(self.m)
        # u *= self.gamma / np.linalg.norm(u)
        return u
    def draw_random_control(self, t):
        u = np.random.randn(self.m)
        # u *= self.gamma / np.linalg.norm(u)
        return u

    def gradient_step(self, x, x_, u):

        z = torch.zeros(self.d + self.m)
        z[:self.d] = torch.tensor(x)
        z[self.d:] = torch.tensor(u)
        prediction = self.net(z)
        y = torch.tensor(x_)
        loss = (prediction-y)@(prediction-y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

