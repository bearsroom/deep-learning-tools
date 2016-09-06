
import math


def build_pred_dict(tag_list, pred_list, thresh):
    """ build prediction dict after filtering by theshold lowerbound
        tag_list format: tag_name
        pred_list format: (im_name, labels, probs)
    """
    pred_dict = {}
    for tag in tag_list:
        pred_dict[tag] = []
    for pred in pred_list:
        im_name = pred[0]
        labels = pred[1]
        probs = pred[2]
        for label, prob in zip(labels, probs):
            if label in tag_list:
                if prob > thresh:
                    pred_dict[label].append(im_name)
    return pred_dict


def recall(pred_dict, gt_dict):
    """ calculate recall
        pred_dict format: key: tag_name, value: list of image name
        gt_dict format: key: tag_name, value: list of image name
    """
    rec = {}
    for tag, gt_list in gt_dict.items():
        pred_list = pred_dict[tag]
        num_gt = len(set(gt_list))
        num_tp = len(set(pred_list) & set(gt_list))
        rec[tag] = num_tp / float(num_gt)
    return rec


def recall_thresh(pred_list, gt_dict, thresh):
    pred_dict = build_pred_dict(gt_dict.keys(), pred_list, thresh)
    return recall(pred_dict, gt_dict)


def recall_all_thresh(pred_list, gt_dict, interval_step=0.1):
    num_intervals = int(math.ceil(1 / float(interval_step)))
    rec_thresh = {}
    for tag in gt_dict.keys():
        rec_thresh[tag] = []
    for idx_interval in range(num_intervals):
        thresh = (idx_interval - 1) * interval_step
        rec = recall_thresh(pred_list, gt_dict, thresh)
        for tag in rec.keys():
            rec_thresh[tag].append(rec[tag])
    return rec_thresh


def precsion(pred_dict, gt_dict):
    """ calculate precision
        pred_dict format: key: tag_name, value: list of image name
        gt_dict format: key: tag_name, value: list of image name
    """
    prec = {}
    for tag, gt_list in gt_dict.items():
        pred_list = pred_dict[tag]
        num_pred = len(set(pred_list))
        num_tp = len(set(pred_list) & set(gt_list))
        prec[tag] = num_tp / float(num_pred)
    return prec


def precision_thresh(pred_list, gt_dict, thresh):
    pred_dict = build_pred_dict(gt_dict.keys(), pred_list, thresh)
    return precision(pred_dict, gt_dict)


def precision_all_thresh(pred_list, gt_dict, interval_step=0.1):
    num_intervals = int(math.ceil(1 / float(interval_step)))
    prec_thresh = {}
    for tag in gt_dict.keys():
        prec_thresh[tag] = []
    for idx_interval in range(num_intervals):
        thresh = (idx_interval - 1) * interval_step
        prec = precision_thresh(pred_list, gt_dict, thresh)
        for tag in prec.keys():
            prec_thresh[tag].append(prec[tag])
    return prec_thresh


def false_positive_rate(pred_dict, gt_dict, num_gt):
    """ calculate false positive rate
        pred_dict format: key: tag_name, value: list of image name
        gt_dict format: key: tag_name, value: list of image name
    """
    fpr = {}
    for tag, gt_list in gt_dict.items():
        pred_list = pred_dict[tag]
        num_neg = num_gt - len(set(pred_list))
        num_tp = len(set(pred_list) & set(gt_list))
        fpr[tag] = num_tp / float(num_neg)
    return fpr


def false_positive_rate_thresh(pred_list, gt_dict, thresh, num_gt):
    pred_dict = build_pred_dict(gt_dict.keys(), pred_list, thresh)
    return false_positive_rate(pred_dict, gt_dict, num_gt)


def false_positive_rate_all_thresh(pred_list, gt_dict, num_gt, interval_step=0.1):
    num_intervals = int(math.ceil(1 / float(interval_step)))
    fpr_thresh = {}
    for tag in gt_dict.keys():
        fpr_thresh[tag] = []
    for idx_interval in range(num_intervals):
        thresh = (idx_interval - 1) * interval_step
        fpr = false_positive_rate_thresh(pred_list, gt_dict, thresh, num_gt)
        for tag in fpr.keys():
            fpr_thresh[tag].append(fpr[tag])
    return fpr_thresh


def f1(pred_dict, gt_dict):
    """ calculate recall, precision and f1 score
        pred_dict format: key: tag_name, value: list of image name
        gt_dict format: key: tag_name, value: list of image name
    """
    rec = {}
    prec = {}
    f1 = {}
    for tag, gt_list in gt_dict.items():
        pred_list = pred_dict[tag]
        num_pred = len(set(pred_list))
        num_gt = len(set(gt_list))
        num_tp = len(set(pred_list) & set(gt_list))
        rec[tag] = num_tp / float(num_gt)
        prec[tag] = num_tp / float(num_pred)
        if rec[tag] + prec[tag] > 0:
            f1[tag] = 2 * rec[tag] * prec[tag] / (rec[tag] + prec[tag])
        else:
            f1[tag] = 0
    return rec, prec, f1


def f1_thresh(pred_list, gt_dict, thresh):
    pred_dict = build_pred_dict(gt_dict.keys(), pred_list, thresh)
    return f1(pred_dict, gt_dict)


def f1_all_thresh(pred_list, gt_dict, interval_step=0.1):
    num_intervals = int(math.ceil(1 / float(interval_step)))
    rec_thresh = {}
    prec_thresh = {}
    f1_score_thresh = {}
    for tag in gt_dict.keys():
        rec_thresh[tag] = []
        prec_thresh[tag] = []
        f1_score_thresh[tag] = []

    for idx_interval in range(num_intervals):
        thresh = idx_interval * interval_step
        rec, prec, f1 = f1_thresh(pred_list, gt_dict, thresh)
        for tag in gt_dict.keys():
            rec_thresh[tag].append(rec[tag])
            prec_thresh[tag].append(prec[tag])
            f1_score_thresh[tag].append(f1[tag])
    return rec_thresh, prec_thresh, f1_score_thresh


