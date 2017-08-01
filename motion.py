#!/usr/bin/python3
# coding: utf-8

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.dataset import convert
import cv2

# data set
class Dataset():
    def __init__(self):
        # 生成画像サイズ
        self.height = 64
        self.width  = 64
        # 画像数
        self.num_img = 3000

    # 画像の生成と保存
    def mk_image(self, name, x, y, radius, color, thickness):
        # 白紙の画像を生成
        img = np.ones( (self.height, self.width), dtype=np.uint8 ) * 255
        # 円を描画
        img = cv2.circle(img, (x, y), radius, color, thickness)
        # 保存
        cv2.imwrite("./circle_motion/{}.png".format(name), img)

        return img

    def make(self):
        # 画像生成処理
        for idx in range(self.num_img+1):
            if( idx % 100 == 0 ):
                print('idx:', idx)
                # パラメータを乱数で決定
                r  = np.random.randint(10)+1
                dx = np.random.randint(-5, 5)
                dy = np.random.randint(-5, 5)
                x  = np.random.randint(r + max(0,-dx*2) + 2, self.width  - r + min(0,-dx*2) - 2 )
                y  = np.random.randint(r + max(0,-dy*2) + 2, self.height - r + min(0,-dy*2) - 2 )
                for frame in range(3):
                    # 画像生成
                    name = 'd{:05}_{}'.format(idx, frame)
                    img = self.mk_image(name, x, y, r, 0, 2)
                    x += dx
                    y += dy

# ネットワーク定義
class Net(chainer.Chain):

    def __init__(self, n_units, n_out, n_channel=32, n_channel2=128, n_channel3=256, loss_function = F.mean_absolute_error):
        super().__init__(
            conv1 = L.Convolution2D(None, n_channel, 5),
            conv2 = L.Convolution2D(None, n_channel2, 3),
            conv3 = L.Convolution2D(None, n_channel3, 3),
            dcnv3 = L.Deconvolution2D(None, n_channel2, 3),
            dcnv2 = L.Deconvolution2D(None, n_channel, 3),
            dcnv1 = L.Deconvolution2D(None, 1, 5),
        )
        self.loss_function = loss_function

    def __call__(self, x):
        if( not self._cpu ):
            x = chainer.cuda.to_gpu( x )
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        d3 = F.relu(self.dcnv3(h3))
        d2 = F.relu(self.dcnv2(d3))
        d1 = F.relu(self.dcnv1(d2))
        return d1

    # calculate loss and accuracy
    def forward(self, x, t):
        if( not self._cpu ):
            t = chainer.cuda.to_gpu( t )
        y = self.__call__(x)
        self.loss = self.loss_function(y, t)
        #self.accuracy = F.accuracy(y, t)
        return self.loss

def make_dataset():
    data = []
    for idx in range(2001):
        img = []
        for frame in range(3):
            # 画像生成
            name = 'd{:05}_{}'.format(idx, frame)
            m = cv2.imread('./circle_motion/{}.png'.format(name), 0)
            img.append( m.astype(np.float) )
        data.append( (np.array([img[0], img[2]], dtype=np.float32), np.array([img[1]], dtype=np.float32)) )
    return data

def save_result(data):
    cv2.imwrite('result.png', data)

def main():
    # get args
    parser = argparse.ArgumentParser(description='ConvolutionNN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--make-dataset', '-m', action='store_const',
                        const=True, default=False,
                        help='make dataset in circle_motion directory')
    args = parser.parse_args()

    if args.make_dataset:
        print('Generate dataset.')
        d = Dataset()
        d.make()
        print('Finish. Saved in circle_motion directory.')
        print('--------')
        print('')

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
    train = make_dataset()
    print("train shape:", train[0][0].shape)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter  = chainer.iterators.SerialIterator(train, args.batchsize)

    # print header
    print("Epoch\tloss(train)")

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
            print("%d\t%f" % (train_iter.epoch, loss.data))

    # 生成した結果を保存
    idx = 0    # データの番号
    x = np.array([train[idx][0]])
    y = model(x)
    if args.gpu >= 0:
        y.to_cpu()
    result = y.data[0][0]
    print(result.shape)
    save_result( result )


if __name__ == '__main__':
    main()

