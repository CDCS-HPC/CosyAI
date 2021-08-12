#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paddle
import numpy as np
from cosyai.trainer.base import _BaseTrainer
from cosyai.util.mapper import activate_functions_mapper

class GradTrainer(_BaseTrainer):
    def __init__(self, conf):
        super().__init__(conf)
        # prepare optimizers
        optimizer_class = activate_functions_mapper(
            self.conf.optimizer or "Adam", paddle.optimizer)
        loss_class = activate_functions_mapper(self.conf.loss or "MSELoss",
                                               paddle.nn)
        metric_class = activate_functions_mapper(
            self.conf.metric or "Accuracy", paddle.metric)

        self.optimizer_class = optimizer_class
        self.loss_class = loss_class
        self.metric_class = metric_class

    def _data_format_check(self, train_set, netname):
        data_shape = len(train_set[0][0])
        if data_shape == 3:
            if netname != 'CNN':
                raise TypeError('Wrong dataformat for {}'.format(netname))

    def train(self, model, train_set, eval_set=None, save_best=True):
        # check data format matching network
        self._data_format_check(train_set, model.netname)

        epochs = self.conf.epochs or 10
        batch_size = self.conf.batch_size or 32
        verbose = self.conf.verbose or 1

        M = paddle.Model(model.net)
        M.prepare(
            optimizer=self.optimizer_class(parameters=model.net.parameters()),
            loss=self.loss_class(),
            metrics=self.metric_class())

        M.fit(train_set,
              epochs=epochs,
              save_dir=self.conf.save_dir,
              batch_size=batch_size,
              verbose=verbose)

        if eval_set is not None:
            M.evaluate(eval_set, verbose=verbose)
        return model

    def test(self, model, test_set):
        verbose = self.conf.verbose or 1
        M = paddle.Model(model.net)
        M.prepare(
            optimizer=self.optimizer_class(parameters=model.net.parameters()),
            loss=self.loss_class(),
            metrics=self.metric_class())
        M.evaluate(test_set, verbose=verbose)
