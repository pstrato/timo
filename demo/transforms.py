# %%
from timo import name, shape

B, C, H, W = name("B"), name("C"), name("H"), name("W")

i = shape(B, (C, 3), H, W)

# %%
from timo import shapes, Transform, Context
from timo.transforms.function import Id, LeakyReLU
from timo.transforms.stop_gradient import StopGradient
from timo.transforms.linear import Linear
from timo.transforms.patch import Patch, square, compbine_patches
from timo.transforms.thread import Thread
from timo.transforms.gaussian import Gaussian

from flax.nnx.rnglib import Rngs
from flax import nnx

from time import monotonic

e = Linear(on=C, to=64, bias=True) >> LeakyReLU() >> Linear(on=C, to=32, bias=True) >> LeakyReLU()
p = StopGradient() >> Patch(on=(H, W), coordinates=compbine_patches(square(1), square(2)), stat="max")
t = Thread(Id(), p, on=C)
# layer = Id(ctx)
ctx = Context(input_shapes=shapes(i), rngs=Rngs(2112))
layer = (e >> t >> Gaussian(on=C, to=2)).transform(ctx)
# layer = e


@nnx.jit
def train(x, model: Transform):
    def loss_fn(model: Transform):
        data = nnx.Dict()
        y = model(x, data)
        return (y**2).mean()

    model.train()
    loss = nnx.value_and_grad(loss_fn)(model)
    return loss


x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (20, 3, 128, 128))
l = train(x, layer)
x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (3, 3, 256, 256))

for _ in range(10):
    start = monotonic()
    l = train(x, layer)
    end = monotonic()
    print(end - start)
l
# %%
from jax import numpy as jnp

covars = jnp.arange(1, 17).reshape((4, 4))
covars_sim = jnp.tril(covars) + jnp.tril(covars, -1).transpose()
