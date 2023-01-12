
from black import Line
import environments.pendulum as pendulum
from active_agents import D_optimal, GradientDesign
from oracles.periodic import Periodic1D
# from models.pendulum import FullLinear



class PeriodicOracle(Periodic1D):

    def __init__(self, model, d, m, gamma, dt):
        super().__init__(model, d, m, gamma, dt)

# class LinearOracle(D_optimal):
#     Model = FullLinear


oracles = {'periodic': PeriodicOracle,
# 'linear': LinearOracle
}
