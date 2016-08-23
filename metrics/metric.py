
import numpy as np

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # all derived class has to implement this method
        pass

    def update(self, pred_int_list, gt_int_list):
        # all derived class has to implement this method
        pass

    def get(self):
        # all derived class has to implement this method
        pass


class Accuracy(Metric):
    def __init__(self, num_classes):
        super(Accuracy, self).__init__(num_classes)
        self.reset():

    def reset(self):
        self.tp = 0
        self.sum = 0

    def update(self, pred_int_list, gt_int_list):
        for y_pred, y_gt in zip(pred_int_list, gt_int_list):
            if y_gt is None:
                continue
            self.sum += 1
            if y_pred == y_gt:
                self.tp += 1

    def get(self):
        if self.sum == 0:
            return None
        else:
            return self.tp / float(self.sum)


class MisClassified(Metric):
    """ save all misclassified images here """
    def __init__(self, num_classes):
        super(MisClassified, self).__init__(num_classes)
        self.reset()

    def reset(self):
        self.fp_images = [[] for _ in range(self.num_classes)]
        self.fn_images = [[] for _ in range(self.num_classes)]

    def update(self, pred_int_list, gt_int_list, probs, im_list):
        for y_pred, y_gt, prob, im_name in zip(pred_int_list, gt_int_list, probs, im_list):
            if y_gt is None:
                continue
            if y_pred != y_gt:
                prob_pred = prob[y_pred]
                prob_gt = prob[y_gt]
                self.fp_images[y_pred].append((im_name, y_gt, prob_pred, prob_gt))
                self.fp_images[y_gt].append((im_name, y_pred, prob_pred, prob_gt))

    def get(self):
        return self.fp_images, self.fn_images


