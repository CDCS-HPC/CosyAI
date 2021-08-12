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

SubConfKey = set(["dataset", "trainer", "model"])


class Config(object):
    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return str(self.d)

    def _get_main_conf(conf, d):
        return {k: v for k, v in d.items() if k not in SubConfKey}

    def __getattr__(self, name):
        if name in SubConfKey:
            d = dict(self.d.get(name, {}), **self._get_main_conf(self.d))
            return self.__class__(d)
        else:
            return self.d.get(name)


def check_config_none(conf, keys=None):
    keys = keys or []
    for key in keys:
        if getattr(conf, key, None) is None:
            raise AttributeError("Config key `{}` should not be None".format(key))