import os, sys, copy
from yaml import load, dump

__all__ = ["get_config"]


class TextboxLayerConfig(object):
    def __init__(self, cfg):
        self._cfg = cfg
        self._def_cfg = cfg['default']
        self._textbox_layer = {}

    def __getitem__(self, key):
        if not isinstance(key, type(0)):
            return self._cfg[key]

        if key in self._textbox_layer:
            return self._textbox_layer[key]

        m = copy.deepcopy(self._def_cfg)

        layer_name = 'layer_' + str(key)

        if layer_name in self._cfg:
            override_cfg = self._cfg[layer_name]
            for k in override_cfg:
                m[k] = override_cfg[k]

        self._textbox_layer[key] = m
        return m

    def __setitem__(self, key, value):
        self._cfg[key] = value


def get_config(config_file):
    with open(config_file) as f:
        config = load(f)

    conf = config['model']['textbox_layer']
    config['model']['textbox_layer'] = TextboxLayerConfig(conf)
    return config

