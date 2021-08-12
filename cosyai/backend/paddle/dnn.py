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

import paddle
from cosyai.model.base import _BaseModel
from cosyai.util.mapper import activate_functions_mapper
from cosyai.util import check_config_none


class DNN(_BaseModel):
    def __init__(self, conf):
        # TODO: Make this variable automatically generated.
        check_config_none(conf, ["input_size", "output_size"])
        hidden_sizes = [conf.input_size] + list(conf.hidden_sizes or [])
        hidden_activations = conf.hidden_activations or (
            ["relu6"] * len(conf.hidden_sizes or []))
        output_activation = conf.output_activation
        self.netname = conf.net

        self.net = _DNNNet(output_size=conf.output_size,
                           hidden_sizes=hidden_sizes,
                           hidden_activations=hidden_activations,
                           output_activation=output_activation)


class _DNNNet(paddle.nn.Layer):
    def __init__(self,
                 output_size,
                 hidden_sizes=None,
                 hidden_activations=None,
                 output_activation=None):
        super().__init__()

        # Default hyper parameters

        hidden_layer_num = len(hidden_sizes)

        self.fcs = [
            paddle.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            for i in range(hidden_layer_num - 1)
        ]
        self.hidden_activations = activate_functions_mapper(
            hidden_activations, paddle.nn.functional)

        self.output_layer = paddle.nn.Linear(hidden_sizes[-1], output_size)

        if output_activation is None:
            self.output_activation = lambda x: x
        else:
            self.output_activation = activate_functions_mapper(
                output_activation, paddle.nn.functional)

    def forward(self, x):
        for fc, act in zip(self.fcs, self.hidden_activations):
            x = act(fc(x))

        return self.output_activation(self.output_layer(x))
