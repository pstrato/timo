from flax.nnx.rnglib import Rngs
from flax import nnx
import jax.numpy as jnp


def test_patch():
    from timo import shape, shapes, TransformContext
    from timo.transforms.patch import Patch, square

    i = shape("H", "W")
    ctx = TransformContext(input_shapes=shapes(i), rngs=Rngs(2112))
    p = Patch(ctx, on=("H", "W"), coordinates=square(1), stat=None)

    x = jnp.array([[1, 2], [3, 4]])
    y = p(x)
    assert y.shape == (2, 2, 8)
    assert jnp.allclose(
        y,
        jnp.array(
            [[[0, 0, 0, 0, 2, 0, 3, 4], [0, 0, 0, 1, 0, 3, 4, 0]], [[0, 1, 2, 0, 4, 0, 0, 0], [1, 2, 0, 3, 0, 0, 0, 0]]]
        ),
    )


def test_patch_max():
    from timo import shape, shapes, TransformContext
    from timo.transforms.patch import Patch, square

    i = shape("H", "W")
    ctx = TransformContext(input_shapes=shapes(i), rngs=Rngs(2112))
    p = Patch(ctx, on=("H", "W"), coordinates=square(1), stat="max")

    x = jnp.array([[1, 2], [3, 4]])
    y = p(x)
    assert y.shape == (2, 2)
    assert jnp.allclose(y, jnp.array([[4, 4], [4, 3]]))


def test_patch_mean():
    from timo import shape, shapes, TransformContext
    from timo.transforms.patch import Patch, square

    i = shape("H", "W")
    ctx = TransformContext(input_shapes=shapes(i), rngs=Rngs(2112))
    p = Patch(ctx, on=("H", "W"), coordinates=square(1), stat="mean")

    x = jnp.array([[1, 2], [3, 4]])
    y = p(x)
    assert y.shape == (2, 2)
    assert jnp.allclose(y, jnp.array([[9 / 3, 8 / 3], [7 / 3, 6 / 3]]))


def test_patch_count():
    from timo.transforms.patch import count, square

    c = count((2, 2), square(1)(("H", "W")))
    assert jnp.allclose(c, jnp.array([[3, 3], [3, 3]]))
