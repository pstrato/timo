# %%
from timo import dim

B, C, H, W = dim("B"), dim("C"), dim("H"), dim("W")

i = B | C * 3 | H | W

# %%
from timo import shapes, TransformContext
from timo.transforms.linear import Linear
from flax.nnx.rnglib import Rngs
from flax import nnx

ctx = TransformContext(input_shapes=shapes(i), rngs=Rngs(2112))
ln1 = Linear(ctx, on=C, to=64, bias=True)
ln2 = Linear(ctx(ln1), on=C, to=32, bias=True)


x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (20, 3, 128, 128))
y = ln2(ln1(x))
x = nnx.nn.initializers.normal(stddev=1)(Rngs(123).input(), (3, 3, 128, 128))
y = ln2(ln1(x))
# %%
