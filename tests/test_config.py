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
