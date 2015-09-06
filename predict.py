# -*- coding: utf-8 -*-

import argparse
from chainer import cuda
import numpy as np
import os
import six.moves.cPickle as pickle

from alexnet import forward
from util import empty_label, load_image, walk_dir

def predict_image(np, model, file_path):
    _, pred = forward(model, load_image(np, file_path), empty_label(np), train=False)
    print '多分これかな？: %s' % (pred.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='pickle file', default='AlexNet_epoch_100.pickle')
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    parser.add_argument('-g', '--gpu', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >=0 else np

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    model = pickle.load(open(args.model, 'rb'))
    if args.gpu >= 0:
        model.to_gpu()

    walk_dir(args.data_dir, lambda _, f: predict_image(xp, model, f))
