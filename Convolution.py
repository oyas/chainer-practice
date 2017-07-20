#!/usr/bin/python3
# coding: utf-8

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.dataset import convert


# ネットワーク定義
class Net(chainer.Chain):

    def __init__(self, n_units, n_out, loss_function = F.softmax_cross_entropy):
        super().__init__(
            conv1=L.Convolution2D(None, 32, 5),
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )
        self.loss_function = loss_function

    def __call__(self, x):
        if( not self._cpu ):
            x = chainer.cuda.to_gpu( x )
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.l2(h1))
        y  = self.l3(h2)
        return y

    def forward(self, x, t):
        if( not self._cpu ):
            x = chainer.cuda.to_gpu( x )
            t = chainer.cuda.to_gpu( t )
        y = self.__call__(x)
        self.loss = self.loss_function(y, t)
        y = self.xp.argmax(y.data, axis=1)
        self.accuracy = sum(y == t) / t.shape[0]
        return self.loss

    def predict(self, x):
        y = self.__call__(x)
        pred = self.xp.argmax(y.data, axis=1)
        return pred


def main():
    # get args
    parser = argparse.ArgumentParser(description='ConvolutionNN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # print config
    print("#GPU       : {}".format(args.gpu))
    print("#batchsize : {}".format(args.batchsize))
    print("#epoch     : {}".format(args.epoch))

    # Model
    model = Net(50, 10)
    # setting for using GPU
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Data
    train, test = chainer.datasets.get_mnist()
    # convert to 3 dim tensor (channel, height, width)
    X_train = [( t[0].reshape(1, 28, 28), t[1] ) for t in train ]
    X_test  = [( t[0].reshape(1, 28, 28), t[1] ) for t in test  ]
    train_iter = chainer.iterators.SerialIterator(X_train, args.batchsize)
    test_iter  = chainer.iterators.SerialIterator(X_test,  args.batchsize, repeat=False, shuffle=False)

    # print header
    print("Epoch\tloss(train)\taccuracy(test)")

    # train
    while train_iter.epoch < args.epoch:

        # next batch data
        train_batch = train_iter.next()
        (x, t) = convert.concat_examples(train_batch)

        # calculate loss and optimize params
        model.cleargrads()
        loss = model.forward(x, t)
        loss.backward()
        optimizer.update()

        # log every epoch
        if( train_iter.is_new_epoch ):

            # calculate accuracy
            (x, t) = convert.concat_examples(test_iter.dataset)
            model.forward( x, t )

            print("%d\t%f\t%f" % (train_iter.epoch, loss.data, model.accuracy))


if __name__ == '__main__':
    main()

