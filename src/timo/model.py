from __future__ import annotations
from typing import TYPE_CHECKING

from flax.nnx import Module

if TYPE_CHECKING:
    from typing import Callable
    from timo.shape import Shape
    from timo.node import Node

    FunctionType = Callable[[*tuple[Shape, ...]], Node | tuple[Node, ...]]


def model(name: str | None = None):
    def decorator(function: FunctionType):
        if name is None:
            model_name = function.__name__
        else:
            model_name = name
        return Model(model_name, function)

    return decorator


class Inputs:
    __slots__ = ["_shapes"]

    def __init__(self, function: FunctionType):
        from inspect import signature
        from timo.shape import Shape

        self._shapes: dict[str, Shape] = {}
        function_signature = signature(function)
        for arg in function_signature.parameters:
            arg_shape = function_signature.parameters[arg].default
            if not isinstance(arg_shape, Shape):
                raise ValueError()
            self._shapes[arg] = arg_shape

    def names(self):
        return self._shapes.keys()

    def __getitem__(self, name: str):
        return self._shapes[name]

    def __len__(self):
        return len(self._shapes)


class Outputs:
    __slots__ = ["_nodes"]

    def __init__(self, function: FunctionType):
        from timo.node import Node

        outputs = function()
        if isinstance(outputs, Node):
            self._nodes = (outputs,)
        elif isinstance(outputs, tuple):
            if len(outputs) == 0:
                raise ValueError()
            for output in outputs:
                if not isinstance(output, Node):
                    raise ValueError()
            self._nodes = outputs
        else:
            raise ValueError()

    def __getitem__(self, index: int):
        return self._nodes[index]

    def __len__(self):
        return len(self._nodes)


class Model(Module):
    __slots__ = ["_name", "_function", "_inputs", "_outputs"]

    def __init__(self, name: str, function: FunctionType):
        self._name = name
        self._function = function
        self._inputs = Inputs(function)
        self._outputs = Outputs(function)

    @property
    def name(self):
        return self._name

    @property
    def function(self):
        return self._function

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __call__(self, *args, **kwargs):
        pass
