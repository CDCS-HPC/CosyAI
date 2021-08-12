#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
