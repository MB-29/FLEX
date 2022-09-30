
import environments.aircraft as aircraft
from agents import Random
from active_agents import D_optimal
from models.aircraft import LinearModel


class LinearDOptimal(D_optimal):
    Model = LinearModel
class LinearRandom(Random):
    Model = LinearModel


oracles = {'linear_D-optimal': LinearDOptimal, 'linear_random':LinearRandom}
