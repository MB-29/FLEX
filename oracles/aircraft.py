
import environments.aircraft as aircraft
from agents import Random
from active_agents import D_optimal
from models.aircraft import FullLinear


class LinearDOptimal(D_optimal):
    Model = FullLinear
class LinearRandom(Random):
    Model = FullLinear


oracles = {'linear_D-optimal': LinearDOptimal, 'linear_random':LinearRandom}
