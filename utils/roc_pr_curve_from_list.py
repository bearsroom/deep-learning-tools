
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
import argparse
from split_data_by_tag import parse_line
from evaluate_from_list import parse_gt_list
from metrics_from_list import false_positive_rate_all_thresh, f1_all_thresh, get_best_point
import numpy as np
import math


def draw_one_curve(subplt, x, y, legend, scatter_points=None):
    subplt.plot(x, y, label=legend)
    if scatter_points:
        for point in scatter_points:
            subplt.scatter(point[0], point[1])
            subplt.annotate('(%.4f, %.4f)' % (point[0], point[1]), xy=point)


def set_subplot(subplt, xlabel, ylabel, title):
    subplt.set_xlabel(xlabel)
    subplt.set_ylabel(ylabel)
    subplt.axis([0, 1.0, 0, 1.0])
    subplt.grid()
    subplt.legend(loc='best')
    subplt.set_title(title)


def draw_curves(tag_list, points, output, title, min_precision=None, min_recall=None):
    """ point format: points[tag] = [(recall, precision, fpr, threshold), ...] """
    num_tags = len(tag_list)
    colormap = plt.cm.gist_ncar

    # plot roc and recall-precision curve
    f, (roc, pr) = plt.subplots(1, 2, figsize=(15, 6))
    roc.set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_tags)])
    pr.set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_tags)])

    for tag in tag_list:
        rec = [p[0] for p in points[tag]]
        prec = [p[1] for p in points[tag]]
        fpr = [p[2] for p in points[tag]]
        draw_one_curve(roc, fpr, rec, tag)
        draw_one_curve(pr, rec, prec, tag)

    set_subplot(roc, 'FPR', 'TPR', 'ROC Curve')
    set_subplot(pr, 'Recall', 'Precision', 'Recall-Precision Curve')
    f.suptitle(title)
    f.savefig(output, format='jpg')
    print('Save curves to {}'.format(output))


def get_points(tag_list, recall, precision, fpr, thresh_step):
    num_intervals = int(math.ceil(1 / float(args.step)))
    points = {}
    for tag in tag_list:
        for rec, prec, fp, thresh_idx in zip(recall[tag], precision[tag], fpr[tag], range(1, num_intervals)):
            if tag not in points.keys():
                points[tag] = [(rec, prec, fp, thresh_idx * thresh_step)]
            else:
                points[tag].append((rec, prec, fp, thresh_idx * thresh_step))
    return points


def parse_args():
    parser = argparse.ArgumentParser(description='Draw ROC curve and recall-precision curve')
    parser.add_argument('--tags', required=True,
                        help='Tag id list, in format tag_id, tag_name, tag_threshold')
    parser.add_argument('--im-list', required=True,
                        help='Image results list, in format im_name, tag_name, tag_prob')
    parser.add_argument('--gt-list', required=True,
                        help='Image ground truth list, in format im_name, tag_id, tag_name')
    parser.add_argument('--step', default=0.1, type=float,
                        help='Threshold step to filter')
    parser.add_argument('--output', required=True,
                        help='Output file')
    parser.add_argument('--title', default=None,
                        help='Output curve title')
    parser.add_argument('--min-precision', default=None, type=float,
                        help='Min precision to find best performance point')
    parser.add_argument('--min-recall', default=None, type=float,
                        help='Min recall to find best performance point')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    tag_list = [t.split()[1] for t in open(args.tags).read().splitlines()]
    tag_list.remove('none')
    print('Tags: {}'.format(', '.join(tag_list)))
    im_list = [parse_line(im) for im in open(args.im_list).read().splitlines()]
    im_list = [im for im in im_list if im]
    print('Total {} images'.format(len(im_list)))
    gt_list = open(args.gt_list).read().splitlines()
    num_gt = len(gt_list)
    gt_dict = parse_gt_list(gt_list, tag_list)

    print('Calculating recall(tpr), precision and fpr for all threshold step...')
    rec, prec, _ = f1_all_thresh(im_list, gt_dict, interval_step=args.step)
    fpr = false_positive_rate_all_thresh(im_list, gt_dict, num_gt, interval_step=args.step)

    points = get_points(tag_list, rec, prec, fpr, args.step)
    draw_curves(tag_list, points, args.output, args.title)

    print('-----------------------------------------------------------------------------')
    num_intervals = int(math.ceil(1 / float(args.step)))
    for tag in rec.keys():
        points = [(r, p, f, thresh) for r, p, f, thresh in zip(rec[tag], prec[tag], fpr[tag], range(1, num_intervals))]
        points = sorted(points, key = lambda p: p[0])
        bp = get_best_point(points, y_lower_bound=args.min_precision)
        print('{:<20}: recall: {:<20}, precision: {:<20}, fpr: {:<20}, threshold: {:<5}'.format(tag, bp[0], bp[1], bp[2], bp[3]*args.step))


