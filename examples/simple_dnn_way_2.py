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

# Please run examples from the root path of this project.

import sys
sys.path.append("../cosyai")

from cosyai.model import Model
from cosyai.dataset import Dataset
from cosyai.trainer import Trainer
from cosyai.util import Config

conf = Config({
    "task": "regression",
    "name": "DemoTrain",
    "backend": "paddle",
    "dataset": {
        # "data_type": "segy",
        # "path": "./data/demo.segy"
        "data_type": "RandSet",
        "input_dim": 1080,
        "output_dim": 1,
        "dataset_size": 20000
    },
    "model": {
        "net": "DNN",
        "input_size": 1080,
        "output_size": 1,
        "hidden_sizes": [64, 32, 32]
    },
    "trainer": {
        "epoch": 20,
        "batch_size": 256,
        "save_dir": "./examples/checkpoints"
    }
})

# dataset = SegyRegressionDataset(conf.dataset)
dataset = Dataset(conf.dataset)
model = Model(conf.model)

trainer = Trainer(conf.trainer)

trainer.train(model, dataset.train_set, eval_set=dataset.eval_set)
trainer.test(model, dataset.test_set)
