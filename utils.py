import numpy as np
from scipy.optimize import brentq

def lstsq_update(prior_estimate, prior_gram, z, y):

    posterior_gram = prior_gram + z[:, None]@z[None, :]
    combination = prior_gram@prior_estimate + y*z
    posterior_estimate = np.linalg.solve(posterior_gram, combination)

    return posterior_estimate, posterior_gram


def greedy_optimal_input(M, A, B, x, gamma):
    """Compute the one-step-ahead optimal design for the estimation of A:
        maximize log det (M + x x^T)
        with x_ = Ax + Bu and u of norm gamma

    :param M: Current moment matrix;
    :type M: size d x d numpy array
    :param A: Dynamics matrix.
    :type A: size d x d numpy array
    :param B: Control matrix.
    :type B: size d x m numpy array
    :param x: Current state.
    :type x: size d numpy array
    :param gamma: gamma**2 is the maximal power.
    :type gamma: float
    :return: optimal input
    :rtype: size m numpy array
    """

    x0 = A @ x
    return linear_D_optimal(M, B, x0, gamma)
def linear_D_optimal(M,  B, v, gamma):
    """Compute the one-step-ahead optimal design for the estimation of A:
        maximize log det (M + x x^T)
        with x_ = v + Bu and u of norm gamma

    :param M: Current moment matrix;
    :type M: size d x d numpy array
    :param A: Dynamics matrix.
    :type A: size d x d numpy array
    :param B: Control matrix.
    :type B: size d x m numpy array
    :param x: Current state.
    :type x: size d numpy array
    :param gamma: gamma**2 is the maximal power.
    :type gamma: float
    :return: optimal input
    :rtype: size m numpy array
    """

    M_inv = np.linalg.inv(M)
    Q = - B.T @ M_inv @ B
    b = B.T @ M_inv @ v
    return minimize_quadratic_sphere(Q, b, gamma)


def minimize_quadratic_sphere(Q, b, gamma):
    """Minimize the corresponding quadratic function over the sphere.

    :return: minimizer
    :rtype: size m numpy array
    """

    eigenvalues, eigenvectors = np.linalg.eig(Q)
    indices = eigenvalues.argsort()
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    if not b.any():
        # print(f'b is zero 0')
        return gamma*(eigenvectors[:, 0])

    beta = eigenvectors.T @ b
    mu_l = -eigenvalues[0] + 0.9*(1/gamma)*abs(beta[0])
    mu_u = -eigenvalues[0] + 1.1*(1/gamma)*(np.linalg.norm(b))

    def func(mu):
        return (beta**2 / (eigenvalues+mu)**2).sum() - gamma**2
    mu = brentq(
        func,
        mu_l,
        mu_u,
    )
    c = beta / (eigenvalues + mu)
    u = eigenvectors @ c
    u = np.real(u)

    return u
