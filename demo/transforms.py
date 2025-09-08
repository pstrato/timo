# %%
from timo import dim

C, H, W = dim("C"), dim("H"), dim("W")

i = C * 3 | H | W

# %%
from timo.transforms.linear import Linear
from timo.transforms.concat import Concat
from timo.transforms.repeat import Repeat
from timo.model import model

o1 = i >> Linear(on=C, to=2) >> Linear(on=C, to=1)
o2 = i >> Linear(on=C, to=4) >> Linear(on=C, to=1)

o3 = Concat(o1, o2, on=C) >> Linear(on=C, to=4)
o4, o5 = Repeat(o3, 2)


B, C, H, W = dim("B"), dim("C"), dim("H"), dim("W")


@model()
def autoencoder(input=B | C * 3 | H | W):
    o = input >> Linear(on=C, to=64)
    return o


print(autoencoder.inputs["input"])
print(autoencoder.outputs[0])
