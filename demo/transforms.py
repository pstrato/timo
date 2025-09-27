# %%
from timo import name, shape

B, C, H, W = name("B"), name("C"), name("H"), name("W")

i = shape(B, (C, 3), H, W)

# %%
from timo import shapes, Transform, Context, Out
from timo.transforms.function import Id, LeakyReLU
from timo.transforms.stop_gradient import StopGradient
from timo.transforms.linear import Linear
from timo.transforms.patch import Patch, square, compbine_patches
from timo.transforms.thread import Thread
from timo.transforms.guassian import Gaussian
from timo.processes.unit_output import UnitOutput

from flax.nnx.rnglib import Rngs
from flax import nnx

e = (
    Linear(on=C, to=64, bias=True) + UnitOutput(on=C, weight=0.1)
    >> LeakyReLU()
    >> Linear(on=C, to=32, bias=True) + UnitOutput(on=C, weight=0.1)
    >> LeakyReLU()
)
p = StopGradient() >> Patch(on=(H, W), coordinates=compbine_patches(square(1), square(2)), stat="max")
t = Thread(Id(), p, on=C)
# layer = Id(ctx)
ctx = Context(input_shapes=shapes(i), rngs=Rngs(2112))
layer = (e >> t >> Gaussian(on=C, to=2)).transform(ctx)
# layer = e


@nnx.jit
def train(x, model: Transform):
    def loss_fn(model: Transform):
        out = Out()
        y = model(x, out=out)
        return (y**2).mean() + out.loss_sum()

    model.train()
    loss = nnx.value_and_grad(loss_fn)(model)
    return loss


x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (20, 3, 128, 128))
l = train(x, layer)
x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (3, 3, 256, 256))
l = train(x, layer)
l
# %%
from jax import numpy as jnp

covars = jnp.arange(1, 17).reshape((4, 4))
covars_sim = jnp.tril(covars) + jnp.tril(covars, -1).transpose()
