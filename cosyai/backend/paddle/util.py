#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
