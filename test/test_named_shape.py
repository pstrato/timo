def test_named_shapes_are_equatable():
    from timo import shape

    s1 = shape(("A",), ("B",))
    assert s1 == shape("A", "B")
    assert s1 != shape("B", "A")
    assert s1 != shape("A")
    assert s1 != shape("A", "B", "C")


def test_named_shapes_can_moveaxis():
    from timo import shape

    s = shape("A")
    assert s.moveaxis(0, 0) == s
    assert s.moveaxis(-1, -1) == s
    s = shape("A", "B")
    assert s.moveaxis(0, 0) == s
    assert s.moveaxis(0, 1) == shape("B", "A")
    assert s.moveaxis(1, 0) == shape("B", "A")
    assert s.moveaxis(-1, -1) == s
    assert s.moveaxis(-1, 0) == shape("B", "A")
    assert s.moveaxis(0, -1) == shape("B", "A")


def test_named_shapes_can_be_resized():
    from timo import shape, size

    s = shape("A")
    assert s.resize(size("A")) == s
    assert s.resize(size("A", 1)) == shape(size("A", 1))
