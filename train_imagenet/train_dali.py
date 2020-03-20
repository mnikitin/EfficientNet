import argparse, time, logging, os, sys, math

import mxnet as mx
from mxnet import gluon, autograd

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from efficientnet_model import get_efficientnet, get_efficientnet_lite


# DALI RECORD PIPELINE
class HybridRecPipe(Pipeline):
    def __init__(self, db_prefix, for_train, input_size, batch_size, num_threads, device_id, num_gpus):
        super(HybridRecPipe, self).__init__(batch_size, num_threads, device_id, seed=12+device_id, prefetch_queue_depth=2)
        self.for_train = for_train
        self.input = ops.MXNetReader(path=[db_prefix + ".rec"], index_path=[db_prefix + ".idx"],
                                     random_shuffle=for_train, shard_id=device_id, num_shards=num_gpus)
        self.resize = ops.Resize(device="gpu", resize_x=input_size, resize_y=input_size)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            output_layout = types.NCHW,
                                            crop = (input_size, input_size),
                                            image_type = types.RGB,
                                            mean = [0.485 * 255,0.456 * 255,0.406 * 255],
                                            std = [0.229 * 255,0.224 * 255,0.225 * 255])
        if self.for_train:
            self.decode = ops.ImageDecoderRandomCrop(device="mixed",
                                                     output_type=types.RGB,
                                                     random_aspect_ratio=[3/4, 4/3],
                                                     random_area=[0.08, 1.0],
                                                     num_attempts=100)
            self.color = ops.ColorTwist(device='gpu')
            self.rng_brightness = ops.Uniform(range=(0.6, 1.4))
            self.rng_contrast = ops.Uniform(range=(0.6, 1.4))
            self.rng_saturation = ops.Uniform(range=(0.6, 1.4))
            self.mirror_coin = ops.CoinFlip(probability=0.5)
        else:
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

    def define_graph(self):
        inputs, labels = self.input(name = "Reader")
        images = self.decode(inputs)
        images = self.resize(images)
        if self.for_train:
            images = self.color(images, brightness=self.rng_brightness(),
                                contrast=self.rng_contrast(), saturation=self.rng_saturation())
            output = self.cmnp(images, mirror=self.mirror_coin())
        else:
            output = self.cmnp(images)
        return [output, labels.gpu()]


def get_rec_data_iterators(train_db_prefix, val_db_prefix, input_size, batch_size, devices):
    num_threads = 2
    num_shards = len(devices)
    train_pipes = [HybridRecPipe(train_db_prefix, True, input_size, batch_size,
                                 num_threads, device_id, num_shards) for device_id in range(num_shards)]
    # Build train pipeline to get the epoch size out of the reader
    train_pipes[0].build()
    print("Training pipeline epoch size: {}".format(train_pipes[0].epoch_size("Reader")))
    # Make train MXNet iterators out of rec pipelines
    dali_train_iter = DALIClassificationIterator(train_pipes, train_pipes[0].epoch_size("Reader"))
    if val_db_prefix:
        val_pipes = [HybridRecPipe(val_db_prefix, False, input_size, batch_size,
                                   num_threads, device_id, num_shards) for device_id in range(num_shards)]
        # Build val pipeline get the epoch size out of the reader
        val_pipes[0].build()
        print("Validation pipeline epoch size: {}".format(val_pipes[0].epoch_size("Reader")))
        # Make val MXNet iterators out of rec pipelines
        dali_val_iter = DALIClassificationIterator(val_pipes, val_pipes[0].epoch_size("Reader"))
    else:
        dali_val_iter = None
    return dali_train_iter, dali_val_iter


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train',
                        help='the training data')
    parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val',
                        help='the validation data')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='',
                        help='path of trainer state to load from.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                        help='name of training log file')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)

    filehandler = logging.FileHandler(opt.logging_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info(opt)

    batch_size = opt.batch_size
    classes = 1000
    num_training_samples = 1281167

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    num_batches = num_training_samples // batch_size

    optimizer = 'nag'
    optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum}

    model_name = opt.model
    if 'lite' in model_name:
        net, input_size = get_efficientnet_lite(model_name, num_classes=classes)
    else:
        net, input_size = get_efficientnet(model_name, num_classes=classes)
    assert input_size == opt.input_size
    if opt.resume_params is not '':
        net.load_parameters(opt.resume_params, ctx=context)

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)

    # Two functions for reading data from record file or raw images
    def get_data_rec(rec_train_prefix, rec_val_prefix, batch_size, devices):
        rec_train_prefix = os.path.expanduser(rec_train_prefix)
        rec_val_prefix = os.path.expanduser(rec_val_prefix)
        input_size = opt.input_size

        def batch_fn(batch):
            data = [b.data[0] for b in batch]
            label = [b.label[0] for b in batch]
            return data, label

        train_data, val_data = get_rec_data_iterators(rec_train_prefix, rec_val_prefix, input_size, batch_size // len(devices), devices)
        return train_data, val_data, batch_fn

    train_data, val_data, batch_fn = get_data_rec(opt.rec_train, opt.rec_val, batch_size, context)

    train_metric = mx.metric.Accuracy()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    save_frequency = opt.save_frequency

    def test(ctx, val_data):
        val_data.reset()
        acc_top1.reset()
        acc_top5.reset()
        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch)
            outputs = [net(X) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return (1-top1, 1-top5)

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        if opt.resume_params is '':
            net.collect_params('.*gamma|.*alpha|.*running_mean|.*running_var').initialize(mx.init.Constant(1), ctx=ctx)
            net.collect_params('.*beta|.*bias').initialize(mx.init.Constant(0.0), ctx=ctx)
            net.collect_params('.*weight').initialize(mx.init.Xavier(), ctx=ctx)

        if opt.no_wd:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, kvstore='local')
        if opt.resume_states is not '':
            trainer.load_states(opt.resume_states)

        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)

        best_val_score = 1

        trainer.set_learning_rate(opt.lr)
        for epoch in range(opt.resume_epoch, opt.num_epochs):
            tic = time.time()
            train_data.reset()
            train_metric.reset()
            btic = time.time()

            if epoch in lr_decay_epoch:
                trainer.set_learning_rate(trainer.learning_rate * lr_decay)
                logger.info('Learning rate has been changed to %f' % trainer.learning_rate)

            for i, batch in enumerate(train_data):
                data, label = batch_fn(batch)
                with autograd.record():
                    outputs = [net(X) for X in data]
                    loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)

                train_metric.update(label, outputs)

                if opt.log_interval and not (i+1)%opt.log_interval:
                    train_metric_name, train_metric_score = train_metric.get()
                    logger.info('Epoch[%d/%d] Batch [%d/%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f'%(
                                epoch, opt.num_epochs, i, num_batches,
                                batch_size*opt.log_interval/(time.time()-btic),
                                train_metric_name, train_metric_score, trainer.learning_rate))
                    btic = time.time()

            train_metric_name, train_metric_score = train_metric.get()
            throughput = int(batch_size * i /(time.time() - tic))

            err_top1_val, err_top5_val = test(ctx, val_data)

            logger.info('[Epoch %d] training: %s=%f'%(epoch, train_metric_name, train_metric_score))
            logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f'%(epoch, throughput, time.time()-tic))
            logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

            if err_top1_val < best_val_score:
                best_val_score = err_top1_val
                net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))
                trainer.save_states('%s/%.4f-imagenet-%s-%d-best.states'%(save_dir, best_val_score, model_name, epoch))

            if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
                net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))
                trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, model_name, epoch))

        if save_frequency and save_dir:
            net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, opt.num_epochs-1))
            trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, model_name, opt.num_epochs-1))

    train(context)

if __name__ == '__main__':
    main()
