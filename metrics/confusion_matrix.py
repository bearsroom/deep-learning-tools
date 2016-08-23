
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from metric import Metric

class ConfusionMatrix(Metric):
    def __init__(self, classes):
        super(ConfusionMatrix, self)__init__(len(classes))
        self.classes = classes

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_int_list, gt_int_list):
        for y_pred, y_gt in zip(pred_int_list, gt_int_list):
            if y_gt is None:
                continue
            self.matrix[y_gt, y_pred] += 1

    def get(self):
        return self.matrix

    def normalize(self):
        p = np.sum(self.matrix, axis=1)
        p[np.where(p == 0)[0]] = 1
        self.matrix = self.matrix / p

    def draw(self, output):
        plt.figure()
        plt.imshow(self.matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix - {} classes'.format(self.num_classes))
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.classes, rotation=90)
        plt.yticks(tick_marks, self.classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predict label')
        plt.savefig(output, format='jpg', quality=80)
        plt.clf()
