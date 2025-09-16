def test_named_axis_are_equatable():
    from timo import name

    n1 = name("A")

    assert n1 == "A"
    assert n1 == name("A")
    assert n1 != "B"
    assert n1 != name("B")
