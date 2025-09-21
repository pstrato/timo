# %%
from timo import name, shape

B, C, H, W = name("B"), name("C"), name("H"), name("W")

i = shape(B, (C, 3), H, W)

# %%
from timo import shapes, Transform, TransformContext
from timo.transforms.id import Id
from timo.transforms.stop_gradient import StopGradient
from timo.transforms.linear import Linear
from timo.transforms.dyntanh import DynTanH
from timo.transforms.patch import Patch, square, compbine_patches
from timo.transforms.sequential import Sequential
from timo.transforms.thread import Thread

from flax.nnx.rnglib import Rngs
from flax import nnx

ctx = TransformContext(input_shapes=shapes(i), rngs=Rngs(2112))
ln1 = Linear(ctx, on=C, to=64, bias=True)
dh1 = DynTanH(ctx(ln1), on=C, bias=True)
ln2 = Linear(ctx(dh1), on=C, to=32, bias=True)
dh2 = DynTanH(ctx(ln2), on=C, bias=True)
e = Sequential(ln1, dh1, ln2, dh2)
p = Sequential(
    StopGradient(ctx(e)), Patch(ctx(e), on=(H, W), coordinates=compbine_patches(square(1), square(2)), stat="max")
)

# layer = Id(ctx)
layer = Sequential(e, Thread(Id(ctx(e)), p, on=C))
# layer = e


@nnx.jit
def train(x, model: Transform):
    def loss_fn(model):
        y = model(x, info=None, out=None)
        return (y**2).mean()

    loss = nnx.value_and_grad(loss_fn)(model)
    return loss


x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (20, 3, 128, 128))
l = train(x, layer)
x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (3, 3, 128, 128))
l = train(x, layer)
# %%
