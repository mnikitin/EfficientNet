# EfficientNet-Gluon
[EfficientNet](https://arxiv.org/abs/1905.11946) Gluon implementation

## ImageNet experiments

### Requirements
Python 3.7 or later with packages:
- `mxnet >= 1.5.0`
- `gluoncv >= 0.6.0`
- `nvidia-dali >= 0.19.0`

### Usage
#### Prepare ImageNet dataset
1. Download and extract dataset following this tutorial:<br/>
https://gluon-cv.mxnet.io/build/examples_datasets/imagenet.html
2. Create mxnet-record files following this turorial:<br/>
https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html#imagerecord-file-for-imagenet

#### Clone this repo
```
git clone https://github.com/mnikitin/EfficientNet.git
cd EfficientNet/train_imagenet
```

#### Train your model
Example of training *efficientnet-b0* with *nvidia-dali data loader* using 4 gpus:
```
IMAGENET_RECORD_ROOT='path/to/imagenet/record/files'
MODEL='efficientnet-b0'
python3 train_imagenet_dali.py --rec-train $IMAGENET_RECORD_ROOT/train --rec-val $IMAGENET_RECORD_ROOT/val --input-size 224 --batch-size 48 --num-gpus 4 --num-epochs 50 --lr 0.1 --lr-decay-epoch 20,30,40 --save-dir params-$MODEL --logging-file params-$MODEL/log.txt --save-frequency 5 --mode hybrid --model $MODEL
```

### Results
TBA
