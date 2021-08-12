#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib
from cosyai.util import check_config_none


class Trainer(object):
    """Model"""

    def __new__(cls, conf):
        check_config_none(conf, ["backend"])
        backend = conf.backend
        module = importlib.import_module('cosyai.backend.' + backend)

        trainer_type = conf.trainer_type or "GradTrainer"
        model_class = getattr(module, trainer_type)
        return model_class(conf)


class _BaseTrainer(object):
    """Base class for cosyai Models"""

    def __init__(self, conf, **kwargs):
        self.conf = conf

    def train(self, model, train_set, eval_set=None):
        raise NotImplementedError()

    def test(self, model, test_set):
        raise NotImplementedError()
    