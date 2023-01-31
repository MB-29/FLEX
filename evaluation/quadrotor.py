import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class GridEvaluation:

    def __init__(self, environment, n_points=20):

        self.environment = environment
        self.plot_system = environment.plot_system

        self.n_points = n_points

        self.v_max = environment.v_max
        interval_v = torch.linspace(-self.v_max, self.v_max, n_points)
        self.grid_vx, self.grid_vy = torch.meshgrid(
            interval_v,
            interval_v,
        )
        self.grid = torch.cat([
            self.grid_vx.reshape(-1, 1),
            self.grid_vy.reshape(-1, 1)
            # grid_u.reshape(-1, 1),
        ], 1)

        self.loss_function = nn.MSELoss()

    def evaluate(self, model, t):
        predictions = model.predict(self.grid.clone())
        truth = self.f_star(self.grid.clone())
        # print(predictions)

        loss = self.loss_function(predictions, truth)
        return loss
    
    def f_star(self, v):
        return -(self.environment.rho/self.environment.mass)*torch.abs(v)*v
         