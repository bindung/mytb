import os, sys
from yaml import load, dump

__all__ = ["get_config"]


def make_object(v):
    if isinstance(v, dict):
        return Config(**v)

    if isinstance(v, type([])):
        return [make_object(o) for o in v]

    return v

class Config(object):
    def __init__(self, **entries):
        self.entries = entries
        for k in entries:
            v = entries[k]
            self.__dict__[k] = make_object(v)


def get_config(config_file):
    with open(config_file) as f:
        data = load(f)

    return Config(**data)


if __name__ == "__main__":
    config = get_config("poc/config.yml")
    print(config.model.textbox_layers[0].conv_size)
