import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, optimizer_hooks
from chainer.training import extensions

from capsules import Capsules


class MnistCapsNet(chainer.Chain):

    def __init__(self):
        super(MnistCapsNet, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 256, ksize=9, stride=1)
            self.conv2 = L.Convolution2D(256, 8 * 32, ksize=9, stride=2)
            self.capsules = Capsules(in_capsules=32 * 6 * 6, in_size=8,
                                     out_capsules=10, out_size=16,
                                     n_dynamic_routing_iter=3)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)

        # Make primary capsules
        h = F.transpose(h, axes=(0, 2, 3, 1))
        h = F.reshape(h, shape=(-1, 32 * 36, 8))

        h = self.capsules(h)
        h = F.sqrt(F.sum(h * h, axis=2))
        return h


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='./result',
                        help='Directory to output the result')
    args = parser.parse_args()

    model = L.Classifier(MnistCapsNet())
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu)
        model.to_gpu()

    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size,
                                                  repeat=True, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(optimizer_hooks.WeightDecay(5e-5))

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.out)

    trainer.extend(extensions.ExponentialShift('alpha', 0.9), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'elapsed_time', 'main/loss', 'main/accuracy',
         'validation/main/loss', 'validation/main/accuracy']
    ))

    trainer.run()


if __name__ == '__main__':
    main()
