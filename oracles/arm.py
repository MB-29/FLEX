
import environments.cartpole as cartpole
from oracles.periodic import Periodic2D


class PeriodicOracle(Periodic2D):

    def __init__(self, x0, m, dynamics, model, gamma, dt):
        super().__init__(x0, m, dynamics, model, gamma, dt, cartpole.period)


oracles = {'periodic': PeriodicOracle}
