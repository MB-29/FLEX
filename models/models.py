import torch
import torch.nn as nn


class Model(nn.Module):
    """Implements the learning model for the dynamics.
    """

    def __init__(self, environment, evaluation=None):
        """
        :param environment: _description_
        :type environment: _type_
        :param evaluation: _description_, defaults to None
        :type evaluation: _type_, optional
        """
        super().__init__()
        self.t_period = environment.period / environment.dt
        self.period = environment.period
        self.B_star = torch.tensor(environment.B_star, dtype=torch.float)
        self.d, self.m = environment.d, environment.m
        self.evaluation = evaluation

        self.linear = False

    def forward(self, z):
        """The dynamics function.

        :param z: state action pair
        :type z: array of size d+m
        :return: velocity dx/dt
        :rtype: array of size d
        """

        raise NotImplementedError
