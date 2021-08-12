#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from cosyai.util import check_config_none


class Dataset(object):
    def __new__(cls, conf):
        check_config_none(conf, ["data_type"])
        if conf.data_type == "segy":
            raise NotImplementedError
        module = importlib.import_module('cosyai.dataset')
        return getattr(module, conf.data_type)(conf)


class _BaseDataset(object):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
