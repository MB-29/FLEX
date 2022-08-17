
import environments.aircraft as aircraft
from agents import Random
from active_agents import Linearized
from models.aircraft import LinearModel


class LinearDOptimal(Linearized):
    Model = LinearModel
class LinearRandom(Random):
    Model = LinearModel


oracles = {'linear_D-optimal': LinearDOptimal, 'linear_random':LinearRandom}
