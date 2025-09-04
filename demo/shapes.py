# %%
from timo import dim

N, M = dim("N"), dim("M")

NM = N + "M"
assert NM == N + M

N64 = N * 64

N64M32 = N * 64 + M * 32
assert N64M32.size == 64 * 32

s = N * 64 | M * 32
s

# %%
C, H, W = dim("C"), dim("H"), dim("W")

C3HW = C * 3 | H | W
C3HW
