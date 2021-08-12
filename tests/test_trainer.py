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
