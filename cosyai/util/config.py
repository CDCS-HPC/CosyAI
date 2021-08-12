#!/usr/bin/env python
# -*- coding: utf-8 -*-

SubConfKey = set(["dataset", "trainer", "model"])


class Config(object):
    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return str(self.d)

    def _get_main_conf(conf, d):
        return {k: v for k, v in d.items() if k not in SubConfKey}

    def __getattr__(self, name):
        if name in SubConfKey:
            d = dict(self.d.get(name, {}), **self._get_main_conf(self.d))
            return self.__class__(d)
        else:
            return self.d.get(name)


def check_config_none(conf, keys=None):
    keys = keys or []
    for key in keys:
        if getattr(conf, key, None) is None:
            raise AttributeError("Config key `{}` should not be None".format(key))