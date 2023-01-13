
import environments.cartpole as cartpole
from oracles.periodic import Periodic2D


class PeriodicOracle(Periodic2D):

    def __init__(self, model, d, m, gamma, **kwargs):
        super().__init__(model, d, m, gamma, **kwargs)


oracles = {'periodic': PeriodicOracle}
