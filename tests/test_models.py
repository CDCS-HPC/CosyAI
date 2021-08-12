#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from cosyai.dataset import RandSet
from cosyai.util import Config
from cosyai.model import Model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.conf_dict = {
            "task": "regression",
            "name": "DemoTrain",
            "backend": "paddle",
            "dataset": {
                "data_type": "random",
                "input_dim": 100,
                "output_dim": 1,
                "dataset_size": 1000
            },
            "model": {
                "net": "DNN",
                "input_size": 100,
                "output_size": 1
            }
        }
        conf = Config(self.conf_dict)

        ds = RandSet(conf.dataset)
        self.X, self.y = ds.train_set.data
        self.tX, self.ty = ds.test_set.data

    def test_model(self):
        conf = Config(self.conf_dict)
        model = Model(conf.model)
        y = model(self.X)
        self.assertListEqual(y.shape, [700, 1])
