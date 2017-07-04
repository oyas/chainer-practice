#!/usr/bin/python3
# coding: utf-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.dataset import convert
from chainer.datasets import tuple_dataset

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


# ネットワーク定義
class AutoEncoder(chainer.Chain):

    def __init__(self, n_in, n_units, n_out, loss_function = F.mean_squared_error):
        super().__init__(
            encoder = L.Linear(n_in, n_units),  # n_in -> n_units
            decoder = L.Linear(None, n_out),  # n_units -> n_out
        )
        self.loss_function = loss_function

    def __call__(self, x, hidden=False):
        h = F.relu(self.encoder(x))
        if hidden:
            return h
        else:
             return F.relu(self.decoder(h))

    def loss(self, x, t):
        y = self.__call__(x)
        return self.loss_function(y, t)

    def predictor(self, x):
        return self.__call__(x)


# 結果を画像として保存する
def plot_mnist_data(data, label, filename, size=(28, 28)):
    plt.axis('off')
    plt.imshow(data.reshape(*size), cmap=cm.gray_r, interpolation='nearest')
    plt.title(label, color='red')
    plt.savefig(filename)


def main():
    # config
    max_epoch = 20
    batchsize = 100
    test_index = 22

    # make result directory
    os.makedirs('result', exist_ok=True)

    # Model
    model = AutoEncoder(784, 64, 784)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # make Dataset
    train, test = chainer.datasets.get_mnist()
    train_oneline = [t[0] for t in train]
    train_twoline = tuple_dataset.TupleDataset(train_oneline, train_oneline)
    train_iter = chainer.iterators.SerialIterator(train_twoline, batchsize)


    # print header
    print("Epoch\tloss(train)")

    # train
    while train_iter.epoch < max_epoch:

        train_batch = train_iter.next()
        (x, t) = convert.concat_examples(train_batch)

        model.cleargrads()
        loss = model.loss(x, t)
        loss.backward()
        optimizer.update()

        # log every epoch
        if( train_iter.is_new_epoch ):

            # loss
            (x, t) = convert.concat_examples( train_iter.dataset )
            loss_train = model.loss( x, t )

            print("%d\t%f" % (train_iter.epoch, loss_train.data))

            # plot predict data
            (x, t) = test[ test_index ]
            data = model.predictor( np.array([x]) ).data
            plot_mnist_data(data, t, 'result/epoch_{}.png'.format(train_iter.epoch))
            # plot hidden node
            data = model( np.array([x]), True ).data
            plot_mnist_data(data, t, 'result/epoch_{}_hidden.png'.format(train_iter.epoch), (8,8))


if __name__ == '__main__':
    main()

