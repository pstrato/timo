def test_sized_named_axis_are_equatable():
    from timo import size

    s1 = size("A")
    assert s1 == ("A", None)
    assert s1 == size("A")
    assert s1 != ("A", 1)
    assert s1 != size("A", 1)
