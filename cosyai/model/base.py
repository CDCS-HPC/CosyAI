#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from cosyai.util import check_config_none


class Model(object):
    """Model"""

    def __new__(cls, conf):
        check_config_none(conf, ["backend", "net"])
        backend = conf.backend
        module = importlib.import_module('cosyai.backend.' + backend)
        model_class = getattr(module, conf.net)
        return model_class(conf)


class _BaseModel(object):
    """Base class for cosyai Models"""

    def __init__(self, conf, **kwargs):
        self.conf = conf

    def __call__(self, *input):
        return self.net.forward(*input)

    def _gradient(self):
        raise NotImplementedError()