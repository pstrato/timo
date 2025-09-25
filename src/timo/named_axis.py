class NamedAxis:
    __slot__ = ["_name"]

    def __init__(self, name: str):
        self._name = name

    def __eq__(self, value):
        if isinstance(value, str):
            return str(self) == value
        if not isinstance(value, NamedAxis):
            return False
        return self._name == value._name

    def __ne__(self, value):
        if isinstance(value, str):
            return self._name != value
        if not isinstance(value, NamedAxis):
            return True
        return self._name != value._name

    def __hash__(self):
        return hash(self._name)

    def __str__(self):
        return self._name

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return self._name


def name(value: str | NamedAxis):
    if isinstance(value, NamedAxis):
        return value
    if isinstance(value, str):
        return NamedAxis(value)
    raise ValueError(f"Expected str or NamedAxis, got `{type(value)}`")
