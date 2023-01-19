
from oracles.periodic import Periodic1D


class PeriodicOracle(Periodic1D):

    def __init__(self, model, d, m, gamma, **kwargs):
        super().__init__(model, d, m, gamma, **kwargs)


oracles = {'periodic': PeriodicOracle}
