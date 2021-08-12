#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import importlib
from cosyai.dataset.base import _BaseDataset
from cosyai.util import check_config_none


class RandSet(_BaseDataset):
    def __init__(self, conf):
        super().__init__(conf)
        check_config_none(conf, ["input_dim", "output_dim", "dataset_size"])
        self._create(conf.task, conf.input_dim, conf.output_dim,
                     conf.dataset_size, conf.split_weight)
        self._transform(conf.backend)

    def _create(self,
                task,
                input_dim,
                output_dim,
                num,
                split_weight,):

        split_weight = split_weight or [7, 1, 2]

        x = np.random.rand(num, 1, input_dim,)

        if task == "classification":
            y = np.random.randint(0, 2, (num, output_dim))
        elif task == "regression":
            y = np.random.rand(num, output_dim)
        else:
            raise NotImplementedError()

        indices = np.cumsum(
            np.asarray(split_weight) * num / np.sum(split_weight), dtype=int)

        self.train_set = x[:indices[0]], y[:indices[0]]
        self.eval_set = x[indices[0]:indices[1]], y[indices[0]:indices[1]]
        self.test_set = x[indices[1]:], y[indices[1]:]

    def _transform(self, backend):
        module = importlib.import_module('cosyai.backend.' + backend +
                                                   '.util')
        self.train_set = module.data_transformer(*self.train_set)
        self.eval_set = module.data_transformer(*self.eval_set)
        self.test_set = module.data_transformer(*self.test_set)
