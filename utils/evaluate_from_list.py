
import os
import sys
import argparse
from split_data_by_tag import parse_line, create_tag_dict


def parse_gt_list(gt_list):
    gt_list = [gt.split() for gt in gt_list]
    gt_dict = {}
    for gt in gt_list:
        im_name = gt[0]
        cls = gt[1]
        if cls not in gt_dict.keys():
            gt_dict[cls] = [im_name]
        else:
            gt_dict[cls].append(im_name)
    return gt_dict


def count_top_k(tag_dict, pred_list, filter_thresh=False, top_k=3):
    im_tag_dict = {}
    for tag in tag_dict.keys():
        im_tag_dict[tag] = []

    for im in pred_list:
        try:
            im_name = im[0]
            tags = im[1][:top_k]
            probs = im[2][:top_k]
            for tag, prob in zip(tags, probs):
                if tag in tag_dict.keys():
                    if filter_thresh and float(prob) > tag_dict[tag]:
                        im_tag_dict[tag].append((im_name, tag, prob, tags))
                    elif not filter_thresh:
                        im_tag_dict[tag].append((im_name, tag, prob, tags))
        except Exception, e:
            continue
    return im_tag_dict


def top_k_recall(gt_dict, pred_dict):
    recall = {}
    for cls, gt_list in gt_dict.items():
        pred_list = pred_dict[cls]
        pred_list = [p[0] for p in pred_list]
        num_p = len(gt_list)
        num_tp = len(set(pred_list) & set(gt_list))
        recall[cls] = num_tp / float(num_p)
    return recall


def top_1_precision(gt_dict, pred_dict):
    precision = {}
    for cls, gt_list in gt_dict.items():
        pred_list = pred_dict[cls]
        pred_list = [p[0] for p in pred_list]
        num_pred = len(pred_list)
        num_tp = len(set(pred_list) & set(gt_list))
        precision[cls] = num_tp / float(num_pred)
    return precision


def top_k_classes(gt_dict, pred_dict):
    top_k_classes = {}
    for cls, gt_list in gt_dict.items():
        tags_count = {}
        pred_list = pred_dict[cls]
        tags_list = [p[-1] for p in pred_list]
        for tags in tags_list:
            for tag in tags:
                if tag not in tags_count.keys():
                    tags_count[tag] = 1
                else:
                    tags_count[tag] += 1
        top_k_classes[cls] = sorted(tags_count, key=tags_count.__getitem__, reverse=True)
    return top_k_classes


def test(im_list_file, gt_list_file, tag_id_list_file, filter_thresh=False, top_k=3):
    im_list = open(im_list_file).read().splitlines()
    im_list = [parse_line(im) for im in im_list]
    num_images = len(im_list)
    print('Total {} images in dataset'.format(num_images))

    tag_id_list = open(tag_id_list_file).read().splitlines()
    tag_id_list = [tag.split() for tag in tag_id_list]

    tag_dict = create_tag_dict(tag_id_list)
    im_tag_dict = count_top_k(tag_dict, im_list, filter_thresh=filter_thresh, top_k=top_k)

    gt_list = open(gt_list_file).read().splitlines()
    gt_dict = parse_gt_list(gt_list)

    recall = top_k_recall(gt_dict, im_tag_dict)
    top_classes = top_k_classes(gt_dict, im_tag_dict)
    if top_k == 1:
        precision = top_1_precision(gt_dict, im_tag_dict)
        print('Top {} recall and precision'.format(top_k))
        for cls, prec in precision.items():
            rec = recall[cls]
            print('Class {:<20}: recall: {:<12}, precision: {:<12}'.format(cls, rec, prec))
    else:
        print('Top {} recall'.format(top_k))
        for cls, rec in recall.items():
            classes = top_classes[cls]
            print('Class {:<20}: top-k-recall: {:<12}, top-k-classes: {}'.format(cls, rec, ','.join(classes[:top_k])))



def parse_args():
    parser = argparse.ArgumentParser(description='Split results list by tags')
    parser.add_argument('--tags', required=True,
                        help='Tag id list, in format tag_id, tag_name, tag_threshold')
    parser.add_argument('--im-list', required=True,
                        help='Image results list, in format im_name, tag_name, tag_prob')
    parser.add_argument('--gt-list', required=True,
                        help='Image ground truth list, in format im_name, tag_id, tag_name')
    parser.add_argument('--filter-thresh', action='store_true',
                        help='Filter results by threshold')
    parser.add_argument('--topk', type=int, default=3,
                        help='Evaluate topk results')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    test(args.im_list, args.gt_list, args.tags, args.filter_thresh, args.topk)
