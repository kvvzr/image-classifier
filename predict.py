# -*- coding: utf-8 -*-

import argparse
import os
import six.moves.cPickle as pickle

from alexnet import forward
from util import empty_label, load_image, walk_dir

def predict_image(model, file_path):
    _, pred = forward(model, load_image(file_path), empty_label(), train=False)
    print '多分これかな？: %s' % (pred.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='pickle file', default='AlexNet_epoch_100.pickle')
    parser.add_argument('-d', '--data_dir', type=str, default='data')
    args = parser.parse_args()

    model = pickle.load(open(args.model, 'rb'))
    walk_dir(args.data_dir, lambda _, f: predict_image(model, f))
