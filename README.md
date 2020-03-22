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
python3 train_dali.py --rec-train $IMAGENET_RECORD_ROOT/train --rec-val $IMAGENET_RECORD_ROOT/val --input-size 224 --batch-size 64 --num-gpus 4 --num-epochs 80 --lr 0.1 --lr-decay-epoch 40,60 --save-dir params-$MODEL --logging-file params-$MODEL/log.txt --save-frequency 5 --mode hybrid --model $MODEL
```

### Results
Code in this repo was used to train *efficientnet-b0* and *efficientnet-lite0* models.</br>
Pretrained params are avaliable (18.8 mb in total = 13.7 mb for *extractor* + 5.1 mb for *classifier*).

<table>
  <tr>
    <th></th>
    <th>err-top1</th>
    <th>err-top5</th>
    <th>pretrained params</th>
  </tr>
  <tr>
    <td>efficientnet-b0</td>
    <td>0.335842</td>
    <td>0.128043</td>
    <td><a href="https://www.dropbox.com/s/l2ehu85vmmj3w5w/0.3358-imagenet-efficientnet-b0-47-best.params?dl=0">dropbox link</a></td>
  </tr>
  <tr>
    <td>efficientnet-lite0</td>
    <td>0.305316</td>
    <td>0.106322</td>
    <td><a href="https://www.dropbox.com/s/fozw7xzaid2vuxp/0.3053-imagenet-efficientnet-lite0-56-best.params?dl=0">dropbox link</a></td>
  </tr>
</table>

**Note** that due to limited computational resources obtained results are worse than in the original paper.</br>
Moreover, *efficientnet-lite0* was trained using more gpus and bigger batch size, so in spite of simpler architecture (relu6 instead of swish) its results are better than for *efficientnet-b0* model.</br>
Anyway, I believe provided pretrained params can serve as a good initialization for your task.

That's how *efficientnet-b0* and *efficientnet-lite0* were trained exactly:</br>
```
MODEL='efficientnet-b0'
python3 train_dali.py --rec-train $IMAGENET_RECORD_ROOT/train --rec-val $IMAGENET_RECORD_ROOT/val --input-size 224 --batch-size 56 --num-gpus 4 --num-epochs 50 --lr 0.1 --lr-decay-epoch 20,30,40 --save-dir params-$MODEL --logging-file params-$MODEL/log.txt --save-frequency 5 --mode hybrid --model $MODEL
```
```
MODEL='efficientnet-lite0'
python3 train_dali.py --rec-train $IMAGENET_RECORD_ROOT/train --rec-val $IMAGENET_RECORD_ROOT/val --input-size 224 --batch-size 72 --num-gpus 6 --num-epochs 60 --lr 0.1 --lr-decay-epoch 20,35,50 --save-dir params-$MODEL --logging-file params-$MODEL/log.txt --save-frequency 5 --mode hybrid --model $MODEL
```
