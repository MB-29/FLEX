
from black import Line
import environments.pendulum as pendulum
from active_agents import D_optimal, GradientDesign
from oracles.periodic import Periodic1D
from models.pendulum import FullLinear



class PeriodicOracle(Periodic1D):

    def __init__(self, x0, m, dynamics, model, gamma, dt):
        super().__init__(x0, m, dynamics, model, gamma, dt, pendulum.period)

class LinearOracle(D_optimal):
    Model = FullLinear


oracles = {'periodic': PeriodicOracle}
