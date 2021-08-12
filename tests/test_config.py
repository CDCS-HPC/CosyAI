#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from cosyai.util import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.d = {
            "task": "regression",
            "name": "DemoTrain",
            "backend": "paddle",
            "dataset": {
                "data_type": "random"
            },
            "model": {
                "net": "DNN",
                "input_size": 1080,
                "output_size": 1,
                "hidden_sizes": [64, 32, 32]
            },
            "trainer": {
                "epoch": 20
            }
        }
    
    def test_config(self):
        conf = Config(self.d)
        self.assertEqual(conf.task, "regression")
        self.assertEqual(conf.model.task, "regression")
        self.assertEqual(conf.trainer.epoch, 20)        
