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
# Author: Jayfu
# ==============================================================================

import paddle
from paddle import nn
from cosyai.model.base import _BaseModel
from cosyai.util import check_config_none


class GRU(_BaseModel):
    def __init__(self, conf):
        check_config_none(conf, ["input_size", "output_size", "hidden_size"])
        num_layers = conf.num_layers or 4
        activate_bidi = conf.activate_bidi or False
        self.netname = conf.net

        self.net = _GRUNET(input_size=conf.input_size,
                         output_size=conf.output_size,
                         hidden_size=conf.hidden_size,
                         num_layers=num_layers,
                         activate_bidi=activate_bidi,
                         )


class _GRUNET(nn.Layer):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 num_layers,
                 activate_bidi="False",
                 **kwargs
                 ):
        super().__init__()

        direction = 'bidirect' if activate_bidi else 'forward'
        fc_hidden_size = hidden_size * 2 if activate_bidi else hidden_size

        self.gru = nn.GRU(input_size=input_size, 
                         hidden_size=hidden_size, 
                         num_layers=num_layers, 
                         direction=direction, 
                         dropout=0.5,
                         **kwargs)

        self.fc = nn.Linear(fc_hidden_size, output_size, )

    def forward(self, input):
        out, _ = self.gru(input)
        out = nn.functional.relu6(out)
        out = self.fc(out)

        return out