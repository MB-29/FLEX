import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class Evaluation:

    def __init__(self, environment, n_points=20):

        self.environment = environment
        self.force = environment.force

        self.n_points = n_points

        self.v_max = environment.v_max
        self.q_max = environment.q_max
        self.center_position = environment.center_position

        interval_q = torch.linspace(-self.v_max, self.v_max, n_points)
        self.grid_qx, self.grid_qy = torch.meshgrid(
            interval_q,
            interval_q,
        )
        self.grid = torch.cat([
            self.grid_qx.reshape(-1, 1),
            self.grid_qy.reshape(-1, 1)
            # grid_u.reshape(-1, 1),
        ], 1)

        self.loss_function = nn.MSELoss()

    def evaluate(self, model, t):
        # print(f'model c = {model.c}')
        # predictions = model.predict(self.grid.clone())
        # truth = self.f_star(self.grid.clone(), t)
        cx, cy = self.center_position(t)
        c_hat = model.c.detach().numpy()
        loss = (1/2) * ((c_hat[0]-cx)**2 + (c_hat[1]-cy)**2)
        # print(predictions)
        # loss = self.loss_function(predictions, truth)

        # self.plot_parameters(model)
        # self.plot_portrait(model.predict)
        return loss

    def f_star(self, q, t):
        cx, cy = self.center_position(t)
        qx, qy = torch.unbind(q, dim=1)
        force = self.force(qx, qy, cx, cy, tensor=True)
        return force

    def plot_portrait(self, f):
        predictions = f(self.grid)

        # plt.xlim((-phi_max, phi_max))
        # plt.ylim((-dphi_max, dphi_max))
        # vectors = predictions.shape(n_points, n_points, n_points, n_points, 2)
        vector_x = predictions[:, 0].reshape(
            self.n_points, self.n_points).detach().numpy()
        vector_y = predictions[:, 1].reshape(
            self.n_points, self.n_points).detach().numpy()
        magnitude = np.sqrt(vector_x.T**2 + vector_y.T**2)
        linewidth = magnitude / magnitude.max()
        plt.streamplot(
            self.grid_qx.numpy().T,
            self.grid_qy.numpy().T,
            vector_x.T,
            vector_y.T,
            color='black',
            density=2.0,
            linewidth=linewidth
        )

    def plot_parameters(self, model):
        c = model.c.detach().numpy()
        plt.scatter(*c)