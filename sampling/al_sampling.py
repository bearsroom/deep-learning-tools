
import os
import sys
sys.path.insert(0, '..')
import numpy as np
import argparse
from utils.split_data_by_tag import parse_line

"""
given candidates X: {(p(y_0|x), p(y_1|x), ... p(y_n|x))}
entropy sampling: pick x : entropy(x) = max(entropy(X)),  entropy(x) = -sum( p(y_i|x) * log(p(y_i|x) ))
margin_sampling: pick x : margin(x) = min(margin(X)), margin(x) = p(y_top1|x) - p(y_top2|x)
min_max_sampling: pick x : max(x) = min(max(x)), max(x) = p(y_top1|x)

"""


def entropy(candidates):
    return -np.sum(probs * np.log(probs), axis=1)


def entropy_sampling(candidates, ratio=0.2):
    """ pick ratio * candidates.shape[0] samples according to their entropy
        candidates: numpy ndarry of shape (n_candidates, n_classes) i
        return indexes of picked samples """
    entropies = entropy(candidates)
    indexes = np.argsort(entropies)[::-1]
    num_pick = int(np.ceil(ratio * candidates.shape[0]))
    return indexes[:num_pick]


def entropy_sampling_each_class(candidates, pred, ratio=0.2):
    """ pick ratio * candidates(pred).shape[0] samples for each class """
    num_classes = int(np.max(pred)) + 1
    indexes = np.empty((0, ), dtype=np.int)
    for class_id in range(num_classes):
        pred_indexes = np.where(pred == class_id)[0]
        if pred_indexes.shape[0] > 1:
            pred_candidates = candidates[pred_indexes]
            pred_indexes = pred_indexes[entropy_sampling(pred_candidates, ratio=ratio)]
        indexes = np.hstack(indexes, pred_indexes)
    return indexes


def margin_top1_top2(candidates):
    assert probs.shape[0] >= 2
    probs_sort = np.sort(probs, axis=1)[:,::-1]
    return probs[:, 0] - probs[:, 1]


def margin_sampling(candidates, ratio=0.2):
    """ pick ratio * candidates.shape[0] samples according to their margin between top1 prob and top2 prob """
    margins = margin_top1_top2(candidates)
    indexes = np.argsort(margins)
    num_pick = int(np.ceil(ratio * candidates.shape[0]))
    return indexes[:num_pick]


def margin_sampling_each_class(candidates, pred, ratio=0.2):
    """ pick ratio * candidates(pred).shape[0] samples for each class """
    num_classes = int(np.max(pred)) + 1
    indexes = np.empty((0, ), dtype=np.int)
    for class_id in range(num_classes):
        pred_indexes = np.where(pred == class_id)[0]
        if pred_indexes.shape[0] > 1:
            pred_candidates = candidates[pred_indexes]
            pred_indexes = pred_indexes[margin_sampling(pred_candidates, ratio=ratio)]
        indexes = np.hstack(indexes, pred_indexes)
    return indexes


def min_max_sampling(candidates, ratio=0.2):
    """ pick ratio * candidates.shape[0] samples according to argmin(max(prob)) for each instance """
    max_probs = np.max(candidates, axis=1)
    indexes = np.argsort(max_probs)
    num_pick = int(np.ceil(ratio * candidates.shape[0]))
    return indexes[:num_pick]


def min_max_sampling_each_class(candidates, pred, ratio=0.2):
    """ pick ratio * candidates(pred).shape[0] samples for each class """
    num_classes = int(np.max(pred)) + 1
    indexes = np.empty((0, ), dtype=np.int)
    for class_id in range(num_classes):
        pred_indexes = np.where(pred == class_id)[0]
        if pred_indexes.shape[0] > 1:
            pred_candidates = candidates[pred_indexes]
            pred_indexes = pred_indexes[min_max_sampling(pred_candidates, ratio=ratio)]
        indexes = np.hstack(indexes, pred_indexes)
    return indexes


def parse_args():
    parser = argparse.ArgumentParser(description='Use active learning technique to sample predicted unlabeled samples for labeling')
    parser.add_argument('--im-list', required=True, type=str,
                        help='Image list with predictions and probs')
    parser.add_argument('--classes', default=None, type=str,
                        help='Classes list with format (class_id, class_name, ...), must be set if --class-constraint is used')
    #parser.add_argument('--output-prefix', required=True, type=str,
    #                    help='Output files\' prefix, output files will be [output_prefix]_[class_name].txt')
    parser.add_argument('--output-file', required=True, type=str,
                        help='Output file name')
    parser.add_argument('--class-constraint', action='store_true',
                        help='If set, will ensure sampling ratio * num_samples for each class')
    parser.add_argument('--ratio', default=0.2, type=float,
                        help='Sampling ratio')
    parser.add_argument('--sampling-method', default='min_max', type=str,
                        help='Active learning sampling method', choices=['min_max', 'margin', 'entropy'])
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    im_list = open(args.im_list).read().splitlines()
    im_list = [parse_line(line) for line in im_list]
    print('Got {} samples'.format(len(im_list)))
    print('Every sample has {} class probs'.format(len(im_list[0][2])))

    if args.class_constraint:
        classes = [c.split()[1] for c in open(args.classes).read().splitlines()]

    if args.sampling_method == 'min_max':
        sampling = min_max_sampling
    elif args.sampling_method == 'margin':
        sampling = margin_sampling
    elif args.sampling_method == 'entropy':
        sampling = entropy_sampling

    f = open(args.output_file, 'w')
    if args.class_constraint:
        for cls in classes:
            cls_im_list = [im for im in im_list if im[1][0] == cls]
            cls_probs = np.ndarray([[float(prob) for prob in im[2]] for im in cls_im_list])
            cls_indexes = sampling(cls_probs, ratio=args.ratio)
            for idx in cls_indexes:
                line = cls_im_list[idx]
                line = '{} labels:{} prob:{}'.format(im[0], ','.join(im[1]), ','.join(im[2]))
                f.write(line+'\n')
    else:
        probs = [[float(prob) for prob in im[2]] for im in im_list[:-1]]
        probs = np.array(probs, dtype=float)
        indexes = sampling(probs, ratio=args.ratio)
        print('Got {} hard samples with method {}'.format(len(indexes), args.sampling_method))
        for idx in indexes:
            line = im_list[idx]
            line = '{} labels:{} prob:{}'.format(line[0], ','.join(line[1]), ','.join(line[2]))
            f.write(line+'\n')


