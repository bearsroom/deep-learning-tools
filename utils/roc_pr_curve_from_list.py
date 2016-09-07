
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


def draw_roc_curve(tag_list, rec_thresh, fpr_thresh, output, title=None, min_recall=None):
    num_tags = len(tag_list)
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_tags)])
    for tag in tag_list:
        recall = rec_thresh[tag]
        fpr = fpr_thresh[tag]
        plt.plot(fpr, recall, label=tag)
        points = [(f, r) for f, r in zip(fpr, recall)]
        points = sorted(points, key = lambda p: p[0])
        best_point = get_best_point(points, y_lower_bound=min_recall)
        plt.scatter(best_point[0], best_point[1])
        plt.annotate('(%.4f, %.4f)' % (best_point[0], best_point[1]), xy=best_point)
    if title:
        plt.title('ROC Curve -' + title)
    else:
        plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.axis([0, 1.0, 0, 1.0])
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(output, format='jpg')
    plt.clf()
    print('Save roc curve to {}'.format(output))


def draw_pr_curve(tag_list, rec_thresh, prec_thresh, output, title=None, min_precision=None):
    num_tags = len(tag_list)
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_tags)])
    for tag in tag_list:
        recall = rec_thresh[tag]
        precision = prec_thresh[tag]
        plt.plot(recall, precision, label=tag)
        points = [(r, p) for r, p in zip(recall, precision)]
        points = sorted(points, key = lambda p: p[0])
        best_point = get_best_point(points, y_lower_bound=min_precision)
        plt.scatter(best_point[0], best_point[1])
        plt.annotate('(%.4f, %.4f)' % (best_point[0], best_point[1]), xy=best_point)
    if title:
        plt.title('Recall-Precision-' + title)
    else:
        plt.title('Recall-Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1.0, 0, 1.0])
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(output, format='jpg')
    plt.clf()
    print('Save recall-precision curve to {}'.format(output))


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
    parser.add_argument('--output-prefix', required=True,
                        help='Output files prefix, files will be [output_prefix]_[pr/roc]_curve.jpg')
    parser.add_argument('--title', default=None,
                        help='Output curve title suffix')
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
    print('Drawing roc curve...')
    draw_roc_curve(tag_list, rec, fpr, args.output_prefix+'_roc_curve.jpg', title=args.title, min_recall=args.min_recall)
    print('Drawing recall-precision curve...')
    draw_pr_curve(tag_list, rec, prec, args.output_prefix+'_pr_curve.jpg', title=args.title, min_precision=args.min_precision)

    num_intervals = int(math.ceil(1 / float(args.step)))
    for tag in rec.keys():
        points = [(r, p, f, thresh) for r, p, f, thresh in zip(rec[tag], prec[tag], fpr[tag], range(1, num_intervals))]
        points = sorted(points, key = lambda p: p[0])
        bp = get_best_point(points, y_lower_bound=args.min_precision)
        print('{:<20}: recall: {:<20}, precision: {:<20}, fpr: {:<20}, threshold: {:<5}'.format(tag, bp[0], bp[1], bp[2], bp[3]*args.step))


