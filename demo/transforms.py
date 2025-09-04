# %%
from timo import dim

C, H, W = dim("C"), dim("H"), dim("W")

i = C * 3 | H | W

# %%
from timo.transforms.linear import Linear
from timo.transforms.concat import Concat
from timo.transforms.repeat import Repeat

o1 = i >> Linear(on=C, to=2) >> Linear(on=C, to=1)
o2 = i >> Linear(on=C, to=4) >> Linear(on=C, to=1)

o3 = Concat(o1, o2, on=C) >> Linear(on=C, to=4)
o4, o5 = Repeat(o3, 2)
