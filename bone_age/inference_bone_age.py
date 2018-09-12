import argparse
import os
import io

from singa import device
from singa import tensor
from singa import autograd
from singa import opt

import pickle

from PIL import Image
import numpy as np


class Block(autograd.Layer):

    def __init__(self, in_filters, out_filters, reps, strides=1, padding=0, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = autograd.Conv2d(in_filters, out_filters,
                                        1, stride=strides, padding=padding, bias=False)
            self.skipbn = autograd.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.layers = []

        filters = in_filters
        if grow_first:
            self.layers.append(autograd.ReLU())
            self.layers.append(autograd.SeparableConv2d(in_filters, out_filters,
                                                        3, stride=1, padding=1, bias=False))
            self.layers.append(autograd.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            self.layers.append(autograd.ReLU())
            self.layers.append(autograd.SeparableConv2d(filters, filters,
                                                        3, stride=1, padding=1, bias=False))
            self.layers.append(autograd.BatchNorm2d(filters))

        if not grow_first:
            self.layers.append(autograd.ReLU())
            self.layers.append(autograd.SeparableConv2d(in_filters, out_filters,
                                                        3, stride=1, padding=1, bias=False))
            self.layers.append(autograd.BatchNorm2d(out_filters))

        if not start_with_relu:
            self.layers = self.layers[1:]
        else:
            self.layers[0] = autograd.ReLU()

        if strides != 1:
            self.layers.append(autograd.MaxPool2d(3, strides, padding + 1))
    
    def dump_params(self, opened_file):
        for layer in self.layers:
            if isinstance(layer, autograd.ReLU) or isinstance(layer, autograd.MaxPool2d):
                pass

            elif isinstance(layer, autograd.BatchNorm2d):
                dump_pytensor(layer.scale, opened_file)
                dump_pytensor(layer.bias, opened_file)
                dump_pytensor(layer.running_mean, opened_file)
                dump_pytensor(layer.running_var, opened_file)

            elif isinstance(layer, autograd.SeparableConv2d):
                dump_pytensor(layer.spacial_conv.W, opened_file)
                dump_pytensor(layer.depth_conv.W, opened_file)
        if self.skip is not None:
            dump_pytensor(self.skip.W, opened_file)
            dump_pytensor(self.skipbn.scale, opened_file)
            dump_pytensor(self.skipbn.bias, opened_file)
            dump_pytensor(self.skipbn.running_mean, opened_file)
            dump_pytensor(self.skipbn.running_var, opened_file)

    def load_params(self, opened_file):
        for layer in self.layers:
            if isinstance(layer, autograd.ReLU) or isinstance(layer, autograd.MaxPool2d):
                pass

            elif isinstance(layer, autograd.BatchNorm2d):
                load_nptensor(layer.scale, opened_file)
                load_nptensor(layer.bias, opened_file)
                load_nptensor(layer.running_mean, opened_file)
                load_nptensor(layer.running_var, opened_file)

            elif isinstance(layer, autograd.SeparableConv2d):
                load_nptensor(layer.spacial_conv.W, opened_file)
                load_nptensor(layer.depth_conv.W, opened_file)
        if self.skip is not None:
            load_nptensor(self.skip.W, opened_file)
            load_nptensor(self.skipbn.scale, opened_file)
            load_nptensor(self.skipbn.bias, opened_file)
            load_nptensor(self.skipbn.running_mean, opened_file)
            load_nptensor(self.skipbn.running_var, opened_file)


    def __call__(self, x):
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            if isinstance(y, tuple):
                y = y[0]
            y = layer(y)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x
        y = autograd.add(y, skip)
        return y

def dump_pytensor(pytensor, file):
    np_tensor=tensor.to_numpy(pytensor)
    pickle.dump(np_tensor, file)

def load_nptensor(pytensor, file):
    np_tensor=pickle.load(file)
    pytensor.copy_from_numpy(np_tensor)

class Xception(autograd.Layer):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        num_classes=1000
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = autograd.Conv2d(1, 32, 3, 2, 0, bias=False)
        self.bn1 = autograd.BatchNorm2d(32)

        self.conv2 = autograd.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = autograd.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, padding=0, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, padding=0, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, padding=0, start_with_relu=True, grow_first=True)

        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = autograd.SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = autograd.BatchNorm2d(1536)

        # do relu here
        self.conv4 = autograd.SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = autograd.BatchNorm2d(2048)

        self.globalpooling = autograd.MaxPool2d(10, 1)
        self.fc = autograd.Linear(2048, num_classes)

        self.linear1 = autograd.Linear(1000,256)
        self.linear2 = autograd.Linear(256,2)

        self.layers_with_params = [self.conv1, self.bn1, self.conv2, self.bn2, self.block1, self.block2, self.block3,
                                   self.block4, self.block5, self.block6, self.block7, self.block8, self.conv3, self.bn3, 
                                   self.conv4, self.bn4, self.fc, self.linear1, self.linear2]
    def dump_params(self, pickle_file):
        with open(pickle_file,'wb') as file:
            for layer in self.layers_with_params:
                if isinstance(layer, autograd.Conv2d):
                    dump_pytensor(layer.W, file)
                    if layer.bias is True:
                        dump_pytensor(layer.b, file)
                
                elif isinstance(layer, autograd.BatchNorm2d):
                    dump_pytensor(layer.scale, file)
                    dump_pytensor(layer.bias, file)
                    dump_pytensor(layer.running_mean, file)
                    dump_pytensor(layer.running_var, file)

                elif isinstance(layer, autograd.SeparableConv2d):
                    dump_pytensor(layer.spacial_conv.W, file)
                    dump_pytensor(layer.depth_conv.W, file)

                elif isinstance(layer, autograd.Linear):
                    dump_pytensor(layer.W, file)
                    if layer.bias is True:
                        dump_pytensor(layer.b, file)

                elif isinstance(layer, Block):
                    layer.dump_params(file)
                else:
                    raise ValueError
        file.close()

    def load_params(self, pickle_file):
        file= open(pickle_file, 'rb')
        for layer in self.layers_with_params:
                if isinstance(layer, autograd.Conv2d):
                    load_nptensor(layer.W, file)
                    if layer.bias is True:
                        load_nptensor(layer.b, file)
                
                elif isinstance(layer, autograd.BatchNorm2d):
                    load_nptensor(layer.scale, file)
                    load_nptensor(layer.bias, file)
                    load_nptensor(layer.running_mean, file)
                    load_nptensor(layer.running_var, file)

                elif isinstance(layer, autograd.SeparableConv2d):
                    load_nptensor(layer.spacial_conv.W, file)
                    load_nptensor(layer.depth_conv.W, file)

                elif isinstance(layer, autograd.Linear):
                    load_nptensor(layer.W, file)
                    if layer.bias is True:
                        load_nptensor(layer.b, file)

                elif isinstance(layer, Block):
                    layer.load_params(file)
                else:
                    raise ValueError
        file.close()

    def features(self, input):
        x = self.conv1(input)

        x = self.bn1(x)
        x = autograd.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = autograd.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = autograd.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = autograd.relu(features)
        x = self.globalpooling(x)
        x = autograd.flatten(x)
        x = self.fc(x)
        return x

    def __call__(self, input):
        x = self.features(input)
        x = self.logits(x)

        x=autograd.relu(x)
        x=self.linear1(x)
        x=autograd.relu(x)
        x=self.linear2(x)

        return x

def load_model(params_file):
	model = Xception()
	model.load_params(params_file)
	return model

def image2array(file, size=299):
    im=Image.open(file)
    im=im.resize((size, size), Image.BILINEAR)
    im=im.convert('L')
    im_array = np.asarray(im).astype(np.float32)
    im_array /= 255
    im_array=np.expand_dims(im_array, 0)
    im_array=np.expand_dims(im_array, 1)
    return im_array

dev = device.create_cuda_gpu()

def predict(img, gender, model, args=None):

    assert gender =='female' or gender == 'male', 'please input gender(female or male)'

    autograd.training=False
    img_array=image2array(img)

    inputs = tensor.Tensor(device=dev, data=img_array, requires_grad=False, stores_grad=False)

    y=model(inputs)

    y_np=tensor.to_numpy(y)[0]
    
    for l in range(len(y_np)):
            if y_np[l] <= 0:
                y_np[l] = 1
            
    prediction={}
    if gender == 'female':
        prediction['predicted bone age'] = float(y_np[0])
    else:
        prediction['predicted bone age'] = float(y_np[1])

    return prediction
