import contextlib


class Session:
    def __init__(self, script_path: str, **names) -> None:
        self.script_path = script_path
        self.names = names

    def root_path(self):
        import os

        return os.path.abspath(os.path.dirname(self.script_path))

    def output_path(self, *paths):
        import os

        names_path = os.path.join(*map(lambda kv: "=".join(map(str, kv)), self.names.items()))
        return os.path.join(self.root_path(), *map(str, paths), names_path)

    def __getitem__(self, key: str):
        return self.names[key]


@contextlib.contextmanager
def session(script_path: str, **names):
    try:
        yield Session(script_path, **names)
    finally:
        pass


def dated_session(script_path: str, **names):
    from datetime import datetime

    now = datetime.now(tz=None).isoformat().replace(":", "-")
    return session(script_path, **names, date=now)
