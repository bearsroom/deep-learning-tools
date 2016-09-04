
"""
Generate sample lists for annotations
will output 2 list:
    - easy samples list: for increasing positive samples, select samples with high confidence
    - hard samples list: for hard samples annotations, select samples using some active learning rules (min max, margin, entropy)
"""

import os
import sys
sys.path.insert(0, sys.path[0]+'/../')
import numpy as np
from sampling.al_sampling import min_max_sampling, margin_sampling, entropy_sampling
from utils.split_data_by_tag import parse_line, create_tag_dict
import argparse


def get_tag_easy_samples(im_list, thresh=None, top_ratio=0.2):
    """ input format: (im_name, label, prob) for 1 label and prob """
    num_images = len(im_list)
    num_top_images = int(top_ratio * num_images) + 1
    im_list = sorted(im_list, key = lambda im: im[-1])[-num_top_images:]
    if thresh and type(thresh) == float:
        im_list = [im for im in im_list if im[-1] > thresh]
    return im_list


def get_hard_samples(im_list, min_max_ratio=0.2, margin_ratio=0.2, entropy_ratio=0.2):
    """ input format: (im_name, labels, probs), labels, probs for top-k predictions """
    im_probs = np.array([im[-1] for im in im_list])
    indexes = np.empty((0, ), dtype=int)
    for ratio, sampling in zip((min_max_ratio, margin_ratio, entropy_ratio), (min_max_sampling, margin_sampling, entropy_sampling)):
        if ratio > 0:
            indexes = np.hstack((indexes, sampling(im_probs, ratio=ratio)))
    indexes = set(indexes.tolist())
    out_list = []
    for idx in indexes:
        out_list.append(im_list[idx])
    return out_list


def generate_tag_im_list(tag_list, im_list, topk=None):
    im_tag_dict = {}
    topk = len(im_list[0][-1]) if topk == None else topk
    for im in im_list:
        topk_tags = im[1][:topk]
        topk_probs = im[2][:topk]
        for tag, prob in zip(topk_tags, topk_probs):
            if tag in tag_list:
                if tag not in im_tag_dict.keys():
                    im_tag_dict[tag] = [(im[0], tag, prob)]
                else:
                    im_tag_dict[tag].append((im[0], tag, prob))
    return im_tag_dict


def generate_easy_samples(tag_list, im_list, output_prefix, topk=1, top_ratio=0.2, tag_thresh_list=None):
    im_tag_dict = generate_tag_im_list(tag_list, im_list, topk=topk)
    tag_thesh_list = [None] * len(tag_list) if tag_thresh_list == None else tag_thresh_list
    for tag, thresh in zip(tag_list, tag_thesh_list):
        tag_im_list = get_tag_easy_samples(im_tag_dict[tag], thresh=thresh, top_ratio=top_ratio)
        output_file = output_prefix + '_easy_{}.txt'.format(tag)
        with open(output_file, 'w') as f:
            for im in tag_im_list:
                f.write('{} {} {}\n'.format(im[0], im[1], im[2]))
        print('Save {} easy samples to {}, tag {}'.format(len(tag_im_list), output_file, tag))


def generate_hard_samples(tag_list, im_list, output_prefix, topk=1, min_max_ratio=0.2, margin_ratio=0.2, entropy_ratio=0.2):
    out_im_list = get_hard_samples(im_list, min_max_ratio=min_max_ratio, margin_ratio=margin_ratio, entropy_ratio=entropy_ratio)
    im_tag_dict = generate_tag_im_list(tag_list, out_im_list, topk=topk)
    for tag in tag_list:
        tag_im_list = im_tag_dict[tag]
        output_file = output_prefix + '_hard_{}.txt'.format(tag)
        with open(output_file, 'w') as f:
            for im in tag_im_list:
                f.write('{} {} {}\n'.format(im[0], im[1], im[2]))
        print('Save {} hard samples to {}, tag {}'.format(len(tag_im_list), output_file, tag))


def parse_args():
    parser = argparse.ArgumentParser(description='Split results list by tags')
    parser.add_argument('--tags', required=True,
                        help='Tag id list, in format tag_id, tag_name, tag_threshold')
    parser.add_argument('--im-list', required=True,
                        help='Image results list, in format im_name, tag_name, tag_prob')
    parser.add_argument('--output-prefix', required=True,
                        help='Output prefix, will output results to [output_prefix]_[easy/hard]_[tag].txt')
    parser.add_argument('--topk', default=1, type=int, help='Took topk results, default value is 1')
    parser.add_argument('--not-filter', action='store_true', help='Not to filter with threshold')
    parser.add_argument('--easy-top-ratio', default=0.2, type=float, help='Top [ratio] easy samples')
    parser.add_argument('--min-max-ratio', default=0.2, type=float, help='Top [ratio] hard samples with lowest min max score')
    parser.add_argument('--entropy-ratio', default=0.2, type=float, help='Top [ratio] hard samples with highest entropy')
    parser.add_argument('--margin-ratio', default=0.2, type=float, help='Top [ratio] hard samples with highest margin score')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # parse tag file
    tag_list = [line.split() for line in open(args.tags).read().splitlines()]
    tag_dict = create_tag_dict(tag_list)
    tag_list = tag_dict.keys()
    tag_probs = [tag_dict[tag] for tag in tag_list] if not args.not_filter else None

    # parse result file
    im_list = open(args.im_list).read().splitlines()
    im_list = [parse_line(im) for im in im_list]
    print('Total {} images in dataset'.format(len(im_list)))

    generate_easy_samples(tag_list, im_list, args.output_prefix, topk=args.topk, top_ratio=args.easy_top_ratio, tag_thresh_list=tag_probs)
    generate_hard_samples(tag_list, im_list, args.output_prefix, topk=args.topk, min_max_ratio=args.min_max_ratio, margin_ratio=args.margin_ratio, entropy_ratio=args.entropy_ratio)
