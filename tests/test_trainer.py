#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from cosyai.dataset import RandSet
from cosyai.util import Config
from cosyai.model import Model
from cosyai.trainer import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.conf = Config({
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
            },
            "trainer": {
                "epochs": 15
            }
        })
        self.ds = RandSet(self.conf.dataset)
        self.model = Model(self.conf.model)

    def test_trainer(self):
        
        trainer = Trainer(self.conf.trainer)
        model = trainer.train(self.model, self.ds.train_set, self.ds.eval_set)
        self.assertNotEqual(model, None)
