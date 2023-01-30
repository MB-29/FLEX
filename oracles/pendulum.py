from oracles.periodic import Periodic1D
# from models.pendulum import FullLinear



class PeriodicOracle(Periodic1D):

    def __init__(self, model, d, m, gamma, dt):
        super().__init__(model, d, m, gamma, dt)

# class LinearOracle(Flex):
#     Model = FullLinear


oracles = {'periodic': PeriodicOracle,
# 'linear': LinearOracle
}
