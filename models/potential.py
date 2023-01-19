import numpy as np
import torch
import torch.nn as nn

from evaluation.potential import Evaluation


class NeuralModel(nn.Module):

    def __init__(self, environment):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(2, 8),
        #     nn.Tanh(),
        #     # nn.Linear(8, 8),
        #     # nn.Tanh(),
        #     nn.Linear(8, 2)
        # )
        self.c = nn.Parameter(torch.tensor([0.1, 0.1]))
        self.c.requires_grad = True

        self.lr = 0.01

        self.d, self.m = environment.d, environment.m
        self.force_model = environment.force
        self.alpha = environment.alpha
        dt = environment.dt

        self.evaluation = Evaluation(environment)

        self.B = (dt/2)*np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            ])
        # self.evaluation = None

    def get_B(self, x):
        return self.B

    def forward(self, z):
        qx, vx, qy, vy, ux, uy = torch.unbind(z, dim=1)
        q = torch.stack((qx, qy), dim=1)
        v = torch.stack((vx, vy), dim=1)
        u = torch.stack((ux, uy), dim=1)
        friction = - self.alpha * v
        force = self.predict(q)
        acceleration = force + friction + u
        ax, ay = torch.unbind(acceleration, dim=1)
        x_dot = torch.stack((vx, ax, vy, ay), dim=1)
        return x_dot

    def predict(self, q):
        cx, cy = self.c[0], self.c[1]
        qx, qy = torch.unbind(q, dim=1)
        return self.force_model(qx, qy, cx, cy, tensor=True)
