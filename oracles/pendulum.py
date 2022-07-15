
from black import Line
import environments.pendulum as pendulum
from active_agents import Linearized, GradientDesign
from oracles.periodic import Periodic1D
from models.pendulum import LinearModel



class PeriodicOracle(Periodic1D):

    def __init__(self, x0, m, dynamics, model, gamma, dt):
        super().__init__(x0, m, dynamics, model, gamma, dt, pendulum.period)

class LinearOracle(Linearized):
    Model = LinearModel


oracles = {'periodic': PeriodicOracle, 'linear': LinearOracle}
