# %%
from timo import name, shape

B, C, H, W = name("B"), name("C"), name("H"), name("W")

i = shape(B, (C, 3), H, W)

# %%
from timo import shapes, TransformModule, TransformContext
from timo.transforms.id import Id
from timo.transforms.stop_gradient import StopGradient
from timo.transforms.linear import Linear
from timo.transforms.dyntanh import DynTanh
from timo.transforms.patch import Patch, square, compbine_patches
from timo.transforms.sequential import Sequential
from timo.transforms.thread import Thread
from timo.transforms.guassian import Gaussian

from flax.nnx.rnglib import Rngs
from flax import nnx

ctx = TransformContext(input_shapes=shapes(i), rngs=Rngs(2112))
ln1 = Linear(ctx, on=C, to=64, bias=True)
dh1 = DynTanh(ln1, on=C, bias=True)
ln2 = Linear(dh1, on=C, to=32, bias=True)
dh2 = DynTanh(ln2, on=C, bias=True)
e = Sequential(ln1, dh1, ln2, dh2)
p = Sequential(StopGradient(e), Patch(e, on=(H, W), coordinates=compbine_patches(square(1), square(2)), stat="max"))
t = Thread(Id(e), p, on=C)
# layer = Id(ctx)
layer = Sequential(e, t, Gaussian(t, on=C, to=2)).module()
# layer = e


@nnx.jit
def train(x, model: TransformModule):
    def loss_fn(model):
        y = model(x)
        return (y**2).mean()

    loss = nnx.value_and_grad(loss_fn)(model)
    return loss


x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (20, 3, 128, 128))
l = train(x, layer)
x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (3, 3, 256, 256))
l = train(x, layer)
l
# %%
