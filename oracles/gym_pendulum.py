
from black import Line
import environments.pendulum as pendulum
from policies import Flex, GradientDesign
from oracles.periodic import Periodic1D
from models.pendulum import FullLinear



class PeriodicOracle(Periodic1D):

    def __init__(self, x0, m, dynamics, model, gamma, dt):
        super().__init__(x0, m, dynamics, model, gamma, dt, pendulum.period)

class LinearOracle(Flex):
    Model = FullLinear


oracles = {'periodic': PeriodicOracle}
