import jax.numpy as jnp


def test_patch():
    from timo import shape, shapes, Context
    from timo.transforms.patch import Patch, square

    i = shape("H", "W")
    ctx = Context(input_shapes=shapes(i))
    p = Patch(on=("H", "W"), coordinates=square(1), stat=None, concat=None)

    x = jnp.array([[1, 2], [3, 4]])
    y = p.transform(ctx)(x)
    assert y.shape == (2, 2, 1, 8)
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [[[0, 0, 0, 0, 2, 0, 3, 4]], [[0, 0, 0, 1, 0, 3, 4, 0]]],
                [[[0, 1, 2, 0, 4, 0, 0, 0]], [[1, 2, 0, 3, 0, 0, 0, 0]]],
            ]
        ),
    )


def test_patch_concat():
    from timo import shape, shapes, Context
    from timo.transforms.patch import Patch, square

    i = shape(("C", 1), "H", "W")
    ctx = Context(input_shapes=shapes(i))
    p = Patch(on=("H", "W"), coordinates=square(1), stat=None, concat="C")

    x = jnp.array([[[1, 2], [3, 4]]])
    y = p.transform(ctx)(x)
    assert y.shape == (8, 2, 2)
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [[0, 0], [0, 1]],
                [[0, 0], [1, 2]],
                [[0, 0], [2, 0]],
                [[0, 1], [0, 3]],
                [[2, 0], [4, 0]],
                [[0, 3], [0, 0]],
                [[3, 4], [0, 0]],
                [[4, 0], [0, 0]],
            ]
        ),
    )


def test_patch_max():
    from timo import shape, shapes, Context
    from timo.transforms.patch import Patch, square

    i = shape("H", "W")
    ctx = Context(input_shapes=shapes(i))
    p = Patch(on=("H", "W"), coordinates=square(1), stat="max", concat=None)

    x = jnp.array([[1, 2], [3, 4]])
    y = p.transform(ctx)(x)
    assert y.shape == (2, 2, 1)
    assert jnp.allclose(y, jnp.array([[[4], [4]], [[4], [3]]]))


def test_patch_max_concat():
    from timo import shape, shapes, Context
    from timo.transforms.patch import Patch, square

    i = shape(("C", 1), "H", "W")
    ctx = Context(input_shapes=shapes(i))
    p = Patch(on=("H", "W"), coordinates=square(1), stat="max", concat="C")

    x = jnp.array([[[1, 2], [3, 4]]])
    y = p.transform(ctx)(x)
    assert y.shape == (1, 2, 2)
    assert jnp.allclose(y, jnp.array([[[4, 4], [4, 3]]]))


def test_patch_mean():
    from timo import shape, shapes, Context
    from timo.transforms.patch import Patch, square

    i = shape("H", "W")
    ctx = Context(input_shapes=shapes(i))
    p = Patch(on=("H", "W"), coordinates=square(1), stat="mean", concat=None)

    x = jnp.array([[1, 2], [3, 4]])
    y = p.transform(ctx)(x)
    assert y.shape == (2, 2, 1)
    assert jnp.allclose(y, jnp.array([[[9 / 3], [8 / 3]], [[7 / 3], [6 / 3]]]))


def test_patch_mean_concat():
    from timo import shape, shapes, Context
    from timo.transforms.patch import Patch, square

    i = shape(("C", 1), "H", "W")
    ctx = Context(input_shapes=shapes(i))
    p = Patch(on=("H", "W"), coordinates=square(1), stat="mean", concat="C")

    x = jnp.array([[[1, 2], [3, 4]]])
    y = p.transform(ctx)(x)
    assert y.shape == (1, 2, 2)
    assert jnp.allclose(y, jnp.array([[[9 / 3, 8 / 3], [7 / 3, 6 / 3]]]))


def test_patch_multi_mean_concat():
    from timo import shape, shapes, Context
    from timo.transforms.patch import Patch, square, compbine_patches

    i = shape(("C", 1), "H", "W")
    ctx = Context(input_shapes=shapes(i))
    p = Patch(on=("H", "W"), coordinates=compbine_patches(square(1), square(1), square(1)), stat="mean", concat="C")

    x = jnp.array([[[1, 2], [3, 4]]])
    y = p.transform(ctx)(x)
    assert y.shape == (3, 2, 2)
    print(y)
    assert jnp.allclose(
        y,
        jnp.array(
            [[[9 / 3, 8 / 3], [7 / 3, 6 / 3]], [[9 / 3, 8 / 3], [7 / 3, 6 / 3]], [[9 / 3, 8 / 3], [7 / 3, 6 / 3]]]
        ),
    )


def test_s3_1():
    from timo.transforms.patch import Patch, square, s3

    segments = s3(1)(("H", "W"))
    assert segments == (
        ((-1, -1), (-1, 0), (0, -1)),
        ((-1, 1), (-1, 0), (0, 1)),
        ((1, -1), (1, 0), (0, -1)),
        ((1, 1), (1, 0), (0, 1)),
    )


def test_s3_2():
    from timo.transforms.patch import Patch, square, s3

    segments = s3(2)(("H", "W"))
    print(segments)
    assert segments == (
        ((-2, -2), (-2, -1), (-1, -2)),
        ((-2, -1), (-2, 0), (-2, 1)),
        ((-2, 2), (-2, 1), (-1, 2)),
        ((-1, -2), (0, -2), (1, -2)),
        ((-1, 2), (0, 2), (1, 2)),
        ((2, -2), (2, -1), (1, -2)),
        ((2, -1), (2, 0), (2, 1)),
        ((2, 2), (2, 1), (1, 2)),
    )
