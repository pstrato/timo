from flax import nnx


class Out(nnx.Pytree):
    def __init__(self, keys: set[str]):
        for key in keys:
            setattr(self, key, nnx.data(None))
        self._keys = nnx.static(keys)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        return setattr(self, key, value)

    def clear(self):
        for key in self._keys:
            setattr(self, key, None)
