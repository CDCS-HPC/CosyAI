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
# Author: JayFu
# ==============================================================================

import os, segyio
import numpy as np
from cosyai.util import check_config_none
import cv2

class Preprocess():
    def __init__(self, conf):
        if conf.data_type != 'segy':
            raise NotImplementedError
        check_config_none(conf,["path", "output_dim", "data_format"])
        self.path = conf.path
        self.output_dim = conf.output_dim
        self.traces = []
        self.labels = []
        self.format = conf.data_format
        self.preprocess = conf.preprocess or ["transformation"]
        self._process()

    def _trans_timeSeq(self):
        for i in range(len(self.traces)):
            self.traces[i] = np.resize(self.traces[i], (1, 751))
            # pass

    def _trans_img(self):
        imgs = []
        temp = 0
        self.traces = np.asarray(self.traces)
        for i in range(100, len(self.traces), 100):
            imgs.append(self.traces[temp: i])
            temp = i+1
        # print(len(imgs), len(imgs[0]))
        for i in range(len(imgs)):
            imgs[i] = [cv2.resize(imgs[i], (224,224)) for j in range(3)]
        self.traces = imgs

    def _read_traces(self):
        for i in os.listdir(self.path):
            fname = os.path.join(self.path,i)
            with segyio.open(fname, ignore_geometry=True) as f:
                self.traces+=f.trace

    def _read_labels(self):
        # self.labels = np.random.rand(len(self.traces), self.output_dim)
        self.labels = np.random.randint(0, 2, (len(self.traces), 1))

    def _process(self):
        self._read_traces()
        for i in self.preprocess:
            print("process", i )
            getattr(self, i)(1)
        # 6.30 fit img data size
        self._read_labels()

    def cleaning(self,i):
        i+=1
        print("data cleaning finished")
        return i

    def integration(self, i):
        i+=1
        print("data integration finished")
        return i

    def saving(self, i):
        i+=1
        print("data saving finished")
        return i

    def transformation(self, i):
        print("data transformation finished")
        if self.format == 'img':
            self._trans_img()
        elif self.format == 'timeSeq':
            self._trans_timeSeq()
        return 1

    def reduction(self, i):
        i+=1
        print("data reduction finished")
        return i
