import torch
import torch.nn as nn

from evaluation.quadrotor import GridEvaluation   

class NeuralModel(nn.Module):

    def __init__(self, environment):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            # nn.Linear(8, 8),
            # nn.Tanh(),
            nn.Linear(8, 2)
        )
        self.lr = 0.02

        self.known_acceleration = environment.known_acceleration
        self.d, self.m = environment.d, environment.m

        self.evaluation = GridEvaluation(environment)

    def transform(self, z):
        return z[:, 1:4:2]


    def forward(self, z):
        dx = torch.zeros_like(z[:, :self.d])
        v = self.transform(z)
        phi = z[:, 4]
        u = z[:, self.d:]
        c_phi, s_phi, ux, uy = torch.cos(phi), torch.sin(phi), u[:, 0], u[:, 1]
        friction = self.predict(v)
        a_x, a_y, a_phi = self.known_acceleration(c_phi, s_phi, ux, uy)
        dx[:, 0] = z[:, 1]
        dx[:, 2] = z[:, 3]
        dx[:, 4] = z[:, 3]
        dx[:, 1] = a_x
        dx[:, 3] = a_y
        dx[:, 5] = a_phi
        dx[:, 1:4:2] += friction
        return dx

    def predict(self, v):
        return self.net(v)
