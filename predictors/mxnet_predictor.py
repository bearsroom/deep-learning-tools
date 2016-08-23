
import frameworks.find_mxnet
import mxnet as mx
import numpy as np
from predictor import Predictor


class MXNetPredictor(Predictor):
    def __init__(self, model_prefix, num_epoch, batch_size, classes, gpu_id=0):
        super(MXNetPredictor, self).__init__(batch_size, classes, gpu_id=gpu_id)
        self.model = mx.model.FeedForward.load(model_prefix, num_epoch, ctx=mx.gpu(gpu_id), numpy_batch_size=batch_size)

    def predict(self, batch):
        self.prob = self.model.predict(batch)
        self.pred = np.argsort(self.prob, axis=1)[:,::-1]
        return self.pred, self.prob


