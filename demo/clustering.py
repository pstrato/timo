# %%
from jax import Array, numpy as jnp
from jax.random import key, split, normal, uniform
from timo.observer import ref
from timo.transform import Transform

C = 5
N = {0: 100, 1: 200, 2: 50, 3: 500, 4: 300}
M = {0: (0, 0), 1: (10, -5), 2: (5, 20), 3: (-10, -10), 4: (-30, 30)}
S = {0: 1, 1: 0.5, 2: 3, 3: 1, 4: 2}
data = []
classes = []
rk = key(1237)
for c in range(C):
    rk, sk = split(rk)
    data.append(normal(sk, (N[c], 2)) * S[c] + jnp.array(M[c]))
    classes.extend(c for _ in range(N[c]))
data = jnp.concat(data, axis=0)

# %%
from timo import name, shape

B, D = name("B"), name("C")
i = shape(B, (D, 2))

# %%
from timo.transforms.linear import Linear
from timo.transforms.function import LeakyReLU
from timo.transforms.gaussian import Gaussian
from timo.transforms.gaussian_activation import GaussianActivation
from timo.transforms.softmax import Softmax
from timo import shapes, Context, Factory
from flax.nnx.rnglib import Rngs
from flax import nnx

f = Linear(on=D, to=4) >> Gaussian(on=D) >> Linear(on=D, to=8) >> Gaussian(on=D, to=C)
t = f.transform(Context(input_shapes=shapes(i), rngs=Rngs(2112)))


# class Model(Factory):

#     def create_transform(self, ctx: Context):
#         encoder = (Linear(on=D, to=16) >> Gaussian(on=D, to=5)).transform(ctx)
#         decoder = Linear(on=D, to=16).transform(encoder.output_ctx)

#         def transform(inputs, data: nnx.Dict, encoder: Transform, decoder: Transform):
#             encoded = encoder(inputs, data) + ref("encoded")
#             decoded = decoder(encoded, data) + ref("decoded")
#             return decoded

#         return Transform[Array, Array](
#             transform, ctx, decoder.output_shapes, data={"encoder": encoder, "decoder": decoder}
#         )


# %%
from timo import (
    fit,
    Keeper,
    StopAfterEpoch,
    Batch,
    ShuffleLoader,
    DataLoader,
    BatchLoader,
    ValueLoss,
    targets,
    outputs,
    rmse,
)
from optax import sgd

batch_size = 100

train_data, eval_data = [], []
for i in range(data.shape[0]):
    point = data[i]
    cls = classes[i]
    cls_targets = jnp.zeros(C).at[C].set(1)
    rk, sk = split(rk)
    if uniform(sk, 1) > 0.9:
        partition = eval_data
    else:
        partition = train_data
    partition.append(Batch(point, cls_targets))

train_loader = BatchLoader(ShuffleLoader(DataLoader(train_data)), batch_size)
eval_loader = BatchLoader(DataLoader(eval_data), batch_size)

loss = ValueLoss(targets, outputs, rmse, "loss/rmse") * 0.0001
training = fit(t, nnx.Optimizer(t, sgd(0.01), wrt=nnx.Param), StopAfterEpoch(10), loss, train_loader, eval_loader)
keeper = Keeper()
training = keeper.keep(training)
for epoch in training:
    print(
        epoch.step,
        "epoch:",
        epoch.epoch,
        "loss:",
        epoch.losses.mean(),
        "epoch time:",
        float(epoch.times["epoch time"].sum),
    )
print(
    "train:",
    keeper.best_train.epoch,
    keeper.best_train.losses.mean(),
    "eval:",
    keeper.best_eval.epoch,
    keeper.best_eval.losses.mean(),
)
# %%
