# Adaptive exploration of nonlinear systems

Implementation of our D-optimal  exploration algorithm along with baselines. Pre-defined environments and models are available in directories `environments` and `models`.

## Demo



## Example

```python
from environments.pendulum import DampedPendulum
from models.pendulum import LinearTheta
from agents import Random
from active_agents import D_optimal

T = 300
dt = 1e-2
environment = DampedPendulum(dt)


environment = Environment()
model = Model(environment)
evaluation = model.evaluation

# agent = Random(
agent = D_optimal(
model,
environment.d,
environment.m,
environment.gamma,
dt=environment.dt
)


z_values, error_values = exploration(environment, agent, T, evaluation)

```
