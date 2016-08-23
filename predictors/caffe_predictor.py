
import frameworks.find_caffe
import caffe
import numpy as np
from predictor import Predictor


class CaffePredictor(Predictor):
    def __init__(self, prototxt, caffemodel, batch_size, classes, gpu_id=0):
        super(CaffePredictor, self).__init__(batch_size, classes, gpu_id=gpu_id)
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        self.model = caffe.Net(prototxt, caffemodel, caffe.TEST)
        # reshape input data
        old_batch_size = self.model.blobs['data'].data.shape[0]
        if old_batch_size != batch_size:
            old_shape = self.model.blobs['data'].data.shape
            new_shape = (batch_size, old_shape[1], old_shape[2], old_shape[3])
            self.model.blobs['data'].reshape(*new_shape)

    def predict(self, batch):
        assert self.model.blobs['data'].data.shape == batch.shape
        self.model.blobs['data'].data[...] = batch
        self.prob = self.model.forward()['prob']
        self.pred = np.argsort(self.prob, axis=1)[:,::-1]
        return self.pred, self.prob

