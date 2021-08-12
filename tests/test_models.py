# Copyright 2021 The ChengduSuperComputingCenter Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Author: chaihua
# ==============================================================================

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
