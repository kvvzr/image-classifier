# -*- coding: utf-8 -*-

from chainer import FunctionSet, Variable
import chainer.functions as F

model = FunctionSet(
    conv1=F.Convolution2D(3, 96, 11, stride=4, pad=1),
    conv2=F.Convolution2D(96, 256, 5, stride=1, pad=2),
    conv3=F.Convolution2D(256, 384, 3, stride=1, pad=1),
    conv4=F.Convolution2D(384, 384, 3, stride=1, pad=1),
    conv5=F.Convolution2D(384, 256, 3, stride=1, pad=1),
    fc6=F.Linear(9216, 4096),
    fc7=F.Linear(4096, 4096),
    fc8=F.Linear(4096, 1)
)

def forward(model, x_data, y_data, train=True):
    x = Variable(x_data, volatile=not train)
    t = Variable(y_data, volatile=not train)

    h = F.relu(model.conv1(x))
    h = F.max_pooling_2d(h, 3, stride=2)
    h = F.local_response_normalization(h)

    h = F.relu(model.conv2(h))
    h = F.max_pooling_2d(h, 3, stride=2)
    h = F.local_response_normalization(h)

    h = F.relu(model.conv3(h))
    h = F.relu(model.conv4(h))
    h = F.relu(model.conv5(h))
    h = F.max_pooling_2d(h, 3, stride=2)

    h = F.dropout(F.relu(model.fc6(h)), train=train, ratio=0.6)
    h = F.dropout(F.relu(model.fc7(h)), train=train, ratio=0.6)
    h = model.fc8(h)

    loss = F.mean_squared_error(t, h)

    return loss, h
