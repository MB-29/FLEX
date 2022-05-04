import numpy as np
import matplotlib.pyplot as plt
import torch 

from agents import Agent

d, m, q = 3, 1, 3

w_2, alpha, b = 1, 0.1, 0.5
theta_star = [w_2, alpha, b]

gamma = 1
T = 100
sigma = 0.1

def dynamics(x, u):
    phi, d_phi = x[0], x[1]
    x_ = np.zeros_like(x)
    # noise = sigma * np.random.randn(d)
    x_[0] = d_phi
    x_[1] = -w_2 * np.sin(phi) - alpha*d_phi -b * u
    x_ += sigma * np.random.randn(d)
    return x_

def f_theta(x, u, theta):

    phi, d_phi = x[0], x[1]

    x_ = torch.zeros_like(x)
    # noise = sigma * np.random.randn(d)
    x_[0] = d_phi
    x_[1] = -theta[0]* np.sin(phi) - theta[1]*d_phi -theta[2] * u
    return x_

x0 = np.zeros(d)

theta0 = np.random.rand(q)
agent = Agent(x0, m, dynamics, f_theta, theta0, gamma)

theta_values = agent.identify(T)



# C_error_values = np.linalg.norm(C_values - C_star, axis=(1, 2))
error_values = np.linalg.norm(theta_values - theta_star, axis=1)

# plt.plot(C_error_values)
plt.plot(error_values)
plt.yscale('log')
plt.show()