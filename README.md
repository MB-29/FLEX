# Greedy online identification of linear dynamic systems

Implementation of our D-optimal  exploration algorithm along with baselines. Pre-defined environments and models are available in directories `environments` and `models`.

## Example

```python
from environments.pendulum import DampedPendulum
from models.pendulum import NeuralModel
from agents import Random
from active_agents import D_optimal

T = 300
dt = 1e-2
environment = DampedPendulum(dt)

model = NeuralModel(environment)

# agent = Random(
agent = D_optimal(
    environment.x0.copy(),
    environment.m,
    environment.dynamics,
    model,
    environment.gamma,
    environment.dt
)

test_values = agent.identify(
    T,
    test_function=environment.test_error
)

```
