#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from cosyai.dataset import RandSet
from cosyai.util import Config


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        conf = Config({
            "task": "regression",
            "name": "DemoTrain",
            "backend": "paddle",
            "dataset": {
                "data_type": "random",
                "input_dim": 100,
                "output_dim": 1,
                "dataset_size": 1000
            }
        })
        data = RandSet(conf.dataset)
        self.assertEqual(len(data.train_set), 700)
