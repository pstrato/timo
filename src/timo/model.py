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
        return ModelFactory(model_name, function)

    return decorator


class Inputs:
    __slots__ = ["_shapes"]

    def __init__(self, function: FunctionType):
        from collections import OrderedDict
        from inspect import signature
        from timo.shape import Shape

        self._shapes: OrderedDict[str, Shape] = OrderedDict()
        function_signature = signature(function)
        for arg in function_signature.parameters:
            default = function_signature.parameters[arg].default
            if not isinstance(default, Shape):
                continue
            self._shapes[arg] = default

    def names(self):
        return self._shapes.keys()

    def __getitem__(self, name: str):
        return self._shapes[name]

    def __len__(self):
        return len(self._shapes)


class ModelFactory:
    __slots__ = ["_name", "_function", "_inputs"]

    def __init__(self, name: str, function: FunctionType):
        self._name = name
        self._function = function
        self._inputs = Inputs(function)

    @property
    def name(self):
        return self._name

    @property
    def function(self):
        return self._function

    @property
    def inputs(self):
        return self._inputs

    def __call__(self, *args, **kwargs):
        return Model(self, *args, **kwargs)


class Arguments:
    __slots__ = ["_arguments"]

    def __init__(self, function: FunctionType, *args, **kwargs):
        from collections import OrderedDict
        from inspect import signature, Parameter
        from timo.shape import Shape

        function_signature = signature(function)
        model_arguments: list[Parameter] = []
        for arg in function_signature.parameters:
            param = function_signature.parameters[arg]
            default = param.default
            if isinstance(default, Shape):
                continue
            model_arguments.append(param)

        self._arguments: OrderedDict[str, object] = OrderedDict()
        for arg in args:
            try:
                model_arg = model_arguments.pop(0)
                self._arguments[model_arg.name] = arg
            except IndexError:
                raise ValueError("Too many positional arguments")
        for arg_name, arg in kwargs.items():
            found = False
            for model_arg in model_arguments:
                if model_arg.name == arg_name:
                    found = True
                    self._arguments[arg_name] = arg
            if not found:
                raise ValueError("Named argument not found")
            model_arguments.remove(model_arg)
        if len(model_arguments) > 0:
            raise ValueError("Some argument not given")

    def names(self):
        return self._arguments.keys()

    def __getitem__(self, name: str):
        return self._arguments[name]

    def __len__(self):
        return len(self._arguments)


class Outputs:
    __slots__ = ["_nodes"]

    def __init__(self, function: FunctionType, *args, **kwargs):
        from timo.node import Node

        outputs = function(*args, **kwargs)
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


class Model:
    __slots__ = ["_factory", "_arguments", "_outputs"]

    def __init__(self, factory: ModelFactory, *args, **kwargs):
        self._factory = factory
        self._arguments = Arguments(factory.function, *args, **kwargs)
        self._outputs = Outputs(factory.function, *args, **kwargs)

    @property
    def name(self):
        return self._factory.name

    @property
    def arguments(self):
        return self._arguments

    @property
    def inputs(self):
        return self._factory.inputs

    @property
    def outputs(self):
        return self._outputs
