
import numpy as np
from metrics import Metric

class F1(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = np.zeros((self.num_classes, ), dtype=np.float32)
        self.fp = np.zeros((self.num_classes, ), dtype=np.float32)
        self.p = np.zeros((self.num_classes, ), dtype=np.float32)

    def update(self, pred_int_list, gt_int_list):
        for y_pred, y_gt in zip(pred_int_list, gt_int_list):
            if y_gt is None:
                continue
            self.p[y_gt] += 1
            if y_pred == y_gt:
                self.tp[y_pred] += 1
            else:
                self.fp[y_pred] += 1

    def get(self):
        recall = np.zeros((self.num_classes), dtype=np.float32)
        precision = np.zeros((self.num_classes), dtype=np.float32)
        f1_score = np.zeros((self.num_classes), dtype=np.float32)
        for idx in range(self.num_classes):
            if self.tp[idx] + self.fp[idx] > 0:
                precision[idx] = self.tp[idx] / float(self.tp[idx] + self.fp[idx])
            if self.p[idx] > 0:
                recall[idx] = self.tp[idx] / float(self.p[idx])
            if precision[idx] + recall[idx] > 0:
                f1_score[idx] = 2 * precision[idx] * recall[idx] / float(precision[idx] + recall[idx])
        return recall, precision, f1_score


class RecallTopK(Metric):
    def __init__(self, num_classes, top_k):
        super(RecallTopK, self)__init__(num_classe)
        self.top_k = top_k

    def reset(self):
        self.tp_topk = np.zeros((self.num_classes, ), dtype=np.float32)
        self.p = np.zeros((self.num_classes, ), dtype=np.float32)
        self.fp_images = [[] for _ in range(self.num_classes)]

    def update(self, pred_int_list, gt_int_list):
        for y_pred, y_gt in zip(pred_int_list, gt_int_list):
            if y_gt is None:
                continue
            assert len(y_pred) == self.top_k
            self.p[y_gt] += 1
            if y_gt in y_pred:
                self.tp_topk[y_gt] += 1

    def get(self):
        recall = np.zeros((self.num_classes, ), dtype=np.float32)
        for idx in range(self.num_classes):
            if self.p[idx] > 0:
                recall[idx] = self.tp_topk[idx] / float(self.p[idx])
        return recall

