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
            l3=L.Linear(None, n_out),  # n_units -> n_out
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
        y = np.argmax(y.data, axis=1)
        accuracy = sum(y == t) / t.shape[0]
        return loss, accuracy


def main():
    # config
    Epoch = 20
    batchsize = 100

    # Model
    model = Net(100, 10)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Data
    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
            repeat=False, shuffle=False)


    # print header
    print("Epoch\tloss(train)\taccuracy(train)\taccuracy(test)")

    # train
    epoch = 0
    for batch in train_iter:

        model.zerograds()
        (x, t) = convert.concat_examples( batch )
        loss = model.loss(x, t)
        loss.backward()
        optimizer.update()

        # log every epoch
        if( train_iter.epoch > epoch ):
            epoch += 1

            # accuracy
            loss_train, accuracy_train  = model.loss_with_accuracy( *train._datasets )
            _         , accuracy_test   = model.loss_with_accuracy( *test._datasets )

            print("%d\t%f\t%f\t%f" % (epoch, loss_train.data, accuracy_train, accuracy_test))

        if( epoch >= Epoch ):
            # finish training
            break


if __name__ == '__main__':
    main()

