from agent import Agent

class Oracle(Agent):

    def __init__(self, x0, m, dynamics, model, gamma, dt, physics):
        super().__init__(x0, m, dynamics, model, gamma, dt)
        self.physics = physics
