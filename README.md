# Adaptive exploration of nonlinear systems

Implementation of our D-optimal  exploration algorithm along with baselines. Pre-defined environments and models are available in directories `environments` and `models`.

## Demo

https://youtu.be/BPD9JQzkraE

## Example

```python
from environments.pendulum import DampedPendulum
from models.pendulum import LinearAB
from policies import Random, Flex

T = 300
dt = 1e-2
environment = DampedPendulum(dt)


environment = Environment()
model = Model(environment)
evaluation = model.evaluation

# agent = Random(
agent = Flex(
model,
environment.d,
environment.m,
environment.gamma,
dt=environment.dt
)


z_values, error_values = exploration(environment, agent, T, evaluation)

```
