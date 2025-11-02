# %%
from jax import numpy as jnp

outputs = jnp.arange(1, 5)
others = [
    [1, 2, 3],
    [0, 2, 3],
    [0, 1, 3],
    [0, 1, 2],
]
others = jnp.array(others)
outputs[others]
