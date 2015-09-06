# -*- coding: utf-8 -*-

import argparse
from chainer import cuda, optimizers
import numpy as np
from progressbar import ProgressBar
import random
import six.moves.cPickle as pickle

from alexnet import forward, model
from util import load_image, num_to_label, walk_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-g', '--gpu', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >=0 else np

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # init optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # load data
    data = []
    walk_dir(args.data_dir, lambda i, f: data.extend([(num_to_label(xp, i), load_image(xp, f))]))

    # learn
    for i in range(args.epoch):
        random.shuffle(data)

        t = 0
        pbar = ProgressBar(len(data))
        for (label, img) in data:

            optimizer.zero_grads()
            loss, acc = forward(model, img, label, train=True)
            loss.backward()
            optimizer.update()

            t += 1
            pbar.update(t)

        print '%s 回繰り返し学習を行った' % (i + 1)

    print 'ヨッシャ！ 学習おわったでｗ'

    # dump model
    pickle.dump(model, open('AlexNet_epoch_%s.pickle' % (args.epoch), 'wb'), -1)
