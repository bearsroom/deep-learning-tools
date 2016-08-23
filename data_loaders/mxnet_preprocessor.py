
import frameworks.find_mxnet
import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
import argparse
import sys
import os
import time
import logging


class MXNetPreprocessor:
    def __init__(self, mean_img_file, crop_mode='center', data_shape=(3, 224, 224)):
        self.mean_img = mx.nd.load(mean_img_file)['mean_img'].asnumpy()
        self.crop_mode = crop_mode
        self.data_shape = data_shape

    def crop_image(self, img):
        # crop image
        if self.crop_mode == 'random':
            short_edge = min(img.shape[:2])
            yy = max(np.random.randint(img.shape[0] - short_edge + 1) - 1, 0)
            xx = max(np.random.randint(img.shape[1] - short_edge + 1) - 1, 0)
            crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
        else:
            if self.crop_mode != 'center':
                logging.warning('Currently we provide only random crop and center crop, use center crop by default')
            short_edge = min(img.shape[:2])
            yy = int((img.shape[0] - short_edge) / 2)
            xx = int((img.shape[1] - short_edge) / 2)
            img = img[yy:yy+short_edge, xx:xx+short_edge]
            return img

    def resize(self, img):
        # resize to data_shape (ch, h, w)
        h, w = self.data_shape[1:]
        return transform.resize(img, (h, w))

    def swapaxes(self, img):
        # swap axes to make image from (224, 224, 3) to (3, 224, 224)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        return img

    def substract_mean(self, img):
        return img - self.mean_img

    def preprocess(self, img):
        img = self.crop_image(img)
        img = self.resize(img)
        # convert to numpy.ndarray
        img = np.asarray(img) * 256
        img = self.swapaxes(img)
        img = self.substract_mean(img)
        return img


