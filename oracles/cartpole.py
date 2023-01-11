
from oracles.periodic import Periodic1D


class PeriodicOracle(Periodic1D):

    def __init__(self, model, d, m, gamma, dt):
        super().__init__(model, d, m, gamma, dt)


oracles = {'periodic': PeriodicOracle}
