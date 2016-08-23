
import os, sys
import numpy as np

class Predictor:
    def __init__(self, batch_size, classes, gpu_id=0):
        self.batch_size = batch_size
        self.classes = classes
        self.gpu_id = gpu_id
        # derived class must load model
        self.model = None

    def predict(self, batch):
        """ derived class must implement this method  """
        pass

    def get_label_prob(self, top_k=1):
        if top_k == 1:
            labels = [self.classes[p[0]] for p in pred]
            probs = np.sort(prob, axis=1)[:,-1]
            return labels, probs
        else:
            labels = [[self.classes[p[i]] for i in range(top_k)] for p in pred]
            probs = np.sort(prob, axis=1)[:,::-1][:,:top_k]
            return labels, probs
