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

from paddle.io import Dataset
import paddle.fluid as fluid

class PdDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = tuple([paddle.to_tensor(d, dtype="float32") for d in data])
    
    def __getitem__(self, index):
        return tuple(d[index] for d in self.data)

    def __len__(self):
        return self.data[0].shape[0]

        
def data_transformer(*data):
    return PdDataset(data)
