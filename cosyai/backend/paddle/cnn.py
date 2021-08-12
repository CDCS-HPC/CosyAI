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

from paddle import nn
import paddle.nn.functional as F
from cosyai.model.base import _BaseModel
from cosyai.util import check_config_none


class CNN(_BaseModel):
    def __init__(self, conf):
        check_config_none(conf, [])
        self.net = _CNNNET()
        self.netname = conf.net


class _CNNNET(nn.Layer):
    def __init__(self,):
        super().__init__()

        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(11, 11))
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3))

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features=160000, out_features=10000)
        self.linear2 = nn.Linear(in_features=10000, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = F.relu(out)

        out = self.flatten(out)
        out = self.linear1(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out