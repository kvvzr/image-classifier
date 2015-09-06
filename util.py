# -*- coding: utf-8 -*-

import numpy as np
import os
from PIL import Image

def load_image(file_path, resize=220):
    img = Image.open(file_path)
    img = img.resize((resize, resize))
    img = np.asarray(img).transpose((2, 0, 1))
    data = np.zeros((1, 3, resize, resize))
    data[0] = img
    return data.astype(np.float32)

def empty_label():
    return np.zeros((1, 1)).astype(np.float32)

def num_to_label(num):
    return np.asarray([[num]]).astype(np.float32)

def walk_dir(dir_name, f):
    i = 0
    for (dir_path, dir_names, file_names) in os.walk(dir_name):
        if dir_names:
            continue

        print 'label %s: %s' % (i, dir_path)

        for file_name in file_names:
            if file_name.startswith('.'):
                continue

            file_path = os.path.join(dir_path, file_name)
            print file_path
            f(i, file_path)

        i += 1
