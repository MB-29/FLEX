
import environments.pendulum as pendulum
from oracles.periodic import Periodic1D


class Oracle(Periodic1D):

    def __init__(self, x0, m, dynamics, model, gamma, dt):
        super().__init__(x0, m, dynamics, model, gamma, dt, pendulum.period)
