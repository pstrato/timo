# %%
from jax import numpy as jnp
from jax.random import PRNGKey, normal

C = 5
N = {0: 100, 1: 200, 2: 50, 3: 500, 4: 300}
M = {0: (0, 0), 1: (10, -5), 2: (5, 20), 3: (-10, -10), 4: (-30, 30)}
S = {0: 1, 1: 0.5, 2: 3, 3: 1, 4: 2}
data = []
classes = []
key = PRNGKey(123)
for c in range(C):
    data.append(normal(key, (N[c], 2)) * S[c] + jnp.array(M[c]))
    classes.extend(c for _ in range(N[c]))
data = jnp.concat(data, axis=0)

# %%
from timo import name, shape

B, D = name("B"), name("C")
i = shape(B, (D, 2))

# %%
from timo.transforms.linear import Linear
from timo.transforms.function import LeakyReLU
from timo.transforms.guassian import Gaussian
from timo.transforms.softmax import Softmax
from timo.processes.unit_output import UnitOutput
from timo import shapes, Context
from flax.nnx.rnglib import Rngs

f = Linear(on=D, to=4) + UnitOutput(on=D, weight=0.1) >> LeakyReLU() >> Gaussian(on=D, to=C) >> Softmax(on=D)
t = f.transform(Context(input_shapes=shapes(i), rngs=Rngs(2112)))

batch_size = 100
