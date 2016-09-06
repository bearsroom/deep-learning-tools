
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
import argparse
from split_data_by_tag import parse_line
from evaluate_from_list import parse_gt_list
from metrics_from_list import false_positive_rate_all_thresh, f1_all_thresh
import numpy as np


def draw_roc_curve(tag_list, rec_thresh, fpr_thresh, output, title=None):
    for tag in tag_list:
        recall = rec_thresh[tag]
        fpr = fpr_thresh[tag]
        plt.plot(fpr, recall, label=tag)
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


def draw_pr_curve(tag_list, rec_thresh, prec_thresh, output, title=None):
    for tag in tag_list:
        recall = rec_thresh[tag]
        precision = prec_thresh[tag]
        plt.plot(recall, precision, label=tag)
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
    draw_roc_curve(tag_list, rec, fpr, args.output_prefix+'_roc_curve.jpg', title=args.title)
    print('Drawing recall-precision curve...')
    draw_pr_curve(tag_list, rec, prec, args.output_prefix+'_pr_curve.jpg', title=args.title)
