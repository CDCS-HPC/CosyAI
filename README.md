# CosyAI

## Introduction

The word `Cosy` has the same meaning as `Ba Shi` in Sichuan dialect. 

## Usage

### All in one `config dict`

```python
conf = Config({
    "task": "regression",
    "name": "DemoTrain",
    "backend": "paddle",
    "dataset": {
        # "data_type": "segy",
        # "path": "./data/demo.segy"
        "data_type": "RandSet",
        "input_dim": 1080,
        "output_dim": 1,
        "dataset_size": 20000,
        "out_type": 
    },
    "model": {
        "net": "DNN",
        "input_size": 1080,
        "output_size": 1,
        "hidden_sizes": [64, 32, 32]
    },
    "trainer": {
        "epoch": 20,
        "batch_size": 256,
        "save_dir": "./examples/checkpoints"
    }
})

log = Executer(conf).execute()

```


### Abstract class

```python

dataset = Dataset(conf.dataset)

model = Model(conf.model)

trainer = Trainer(conf.trainer)
trainer.train(model, dataset.train_set, eval_set=dataset.eval_set)
trainer.test(model, dataset.test_set)

```