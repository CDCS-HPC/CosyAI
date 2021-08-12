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

import importlib
from cosyai.util import check_config_none


class Model(object):
    """Model"""

    def __new__(cls, conf):
        check_config_none(conf, ["backend", "net"])
        backend = conf.backend
        module = importlib.import_module('cosyai.backend.' + backend)
        model_class = getattr(module, conf.net)
        return model_class(conf)


class _BaseModel(object):
    """Base class for cosyai Models"""

    def __init__(self, conf, **kwargs):
        self.conf = conf

    def __call__(self, *input):
        return self.net.forward(*input)

    def _gradient(self):
        raise NotImplementedError()