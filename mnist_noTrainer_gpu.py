#!/usr/bin/python3
# coding: utf-8

import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from chainer.dataset import convert


# ネットワーク定義
class Net(chainer.Chain):

    def __init__(self, n_units, n_out, loss_function = F.softmax_cross_entropy):
        super().__init__(
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),    # n_units -> n_out
        )
        self.loss_function = loss_function

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

    def loss(self, x, t):
        y = self.__call__(x)
        return self.loss_function(y, t)

    def loss_with_accuracy(self, x, t):
        y = self.__call__(x)
        loss = self.loss_function(y, t)
        accuracy = F.accuracy(y, t)
        return loss, accuracy


def main():
    # config
    max_epoch = 20
    batchsize = 100
    gpu_no    = 0

    # Model
    model = Net(100, 10)
    # setting for using GPU
    chainer.cuda.get_device(gpu_no).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

    # Optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Data
    train, test = chainer.datasets.get_mnist()
    #train = [ (chainer.cuda.to_gpu(t[0]), chainer.cuda.to_gpu(t[1])) for t in train ]
    #test  = [ (chainer.cuda.to_gpu(t[0]), chainer.cuda.to_gpu(t[1])) for t in test ]
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

    # print header
    print("Epoch\tloss(train)\taccuracy(test)")

    # train
    while train_iter.epoch < max_epoch:

        train_batch = train_iter.next()
        (x, t) = convert.concat_examples(train_batch)

        # copy to GPU
        x = chainer.cuda.to_gpu( x )
        t = chainer.cuda.to_gpu( t )

        # calculate loss and optimize params
        model.cleargrads()
        loss = model.loss(x, t)
        loss.backward()
        optimizer.update()

        # log every epoch
        if( train_iter.is_new_epoch ):

            # calculate accuracy
            (x, t) = convert.concat_examples(test_iter.dataset)
            x = chainer.cuda.to_gpu( x )
            t = chainer.cuda.to_gpu( t )
            _         , accuracy_test   = model.loss_with_accuracy( x, t )

            print("%d\t%f\t%f" % (train_iter.epoch, loss.data, accuracy_test.data))


if __name__ == '__main__':
    main()

