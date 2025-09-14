# %%
from timo import name, size, shape

N, M = name("N"), name("M")

N64 = size(N, 64)

s = shape(size(N, 64), size(M, 32))
s

# %%
C, H, W = name("C"), name("H"), name("W")

C3HW = shape(size(C, 3), size(H), size(W))
C3HW
# %%
C3HW.moveaxis(C, -2)
