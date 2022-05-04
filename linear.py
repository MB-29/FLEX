import numpy as np
import matplotlib.pyplot as plt

from agents import Agent

A_star = np.array([
    [0.9, 1],
    [0, 0.9]
])

B_star = np.array([
    [0],
    [1]
])
# B = np.eye(2, 2)

d, m = B_star.shape
q = d * (d+m)

C_star = np.zeros((d, d+m))
C_star[:, :d] = A_star
C_star[:, d:] = B_star

theta_star = C_star.reshape(-1)

gamma = 1
T = 100
sigma = 0.1

def dynamics(x, u):
    noise = sigma * np.random.randn(d)
    return A_star@x + B_star@u + noise


def f_theta(x, u, theta):

    d = x.shape[0]
    C = theta.view((d, -1))

    A = C[:, :d]
    B = C[:, d:]

    return A@x + B@u


x0 = np.zeros(d)

theta0 = np.zeros(q)
agent = Agent(x0, m, dynamics, f_theta, theta0, gamma)

theta_values = agent.identify(T)
C_values = agent.C_values

error_values = np.linalg.norm(theta_values - theta_star, axis=1)
C_error_values = np.linalg.norm(C_values - C_star, axis=(1, 2))

plt.plot(error_values)
plt.plot(C_error_values)
plt.yscale('log')
plt.show()