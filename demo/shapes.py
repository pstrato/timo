# %%
from timo import dim, shape

N, M = dim("N"), dim("M")

NM = N + "M"
assert NM == N + M

N64 = N * 64

N64M32 = N * 64 + M * 32
assert N64M32.size == 64 * 32

s = shape(N * 64, M * 32)
