"""
Parse records of format EntriesV1, EntriesV2, EntriesV3 and split into training and test set
Input of EntriesV1: volume_id, orginal_url(fid), value1
Input of EntriesV2/EntriesV3: volume_id, entry_id, original_url(fid), tag_id, pos_or_neg
return: multiple lists [output_prefix]_[train, test]_[pos, neg, drop].txt
format of output: volume_id, fid
"""

import os
import argparse
import sys
import random


def ParseLineEntriesV1(entries):
    # format: volume_id, original_url, value1
    meta_pos = []
    meta_neg = []
    meta_drop = []
    for line in entries:
        vid, fid, value1 = line.split(',')
        vid = vid.strip('"')
        fid = fid.strip('"')
        value1 = int(value1.strip('"'))
        if value1 == 1:
            meta_drop.append((vid, fid))
        elif value1 == 2:
            meta_neg.append((vid, fid))
        else:
            meta_pos.append((vid, fid))
    return meta_pos, meta_neg, meta_drop


def ParseLineEntriesV2(entries):
    # format: volume_id, entry_id, original_url, tag_id, pos_or_neg
    meta_pos = []
    meta_neg = []
    meta_drop = []
    for line in entries:
        if ',' in line:
            vid, _, fid, tid, value = line.split(',')
        else:
            vid, _, fid, tid, value = line.split()
        fid = fid.strip('"')
        if 'http' in fid:
            continue
        vid = vid.strip('"')
        value = int(value.strip('"'))
        if value == -1:
            meta_drop.append((vid, fid))
        elif value == 0:
            meta_neg.append((vid, fid))
        elif value == 1:
            meta_pos.append((vid, fid))
    return meta_pos, meta_neg, meta_drop


def save(entries, output_file):
    with open(output_file, 'w') as f:
         for e in entries:
             f.write('{} {}\n'.format(e[0], e[1]))


def parse_file(entries, output_prefix, version='V1'):
    if version == 'V1':
        meta_pos, meta_neg, meta_drop = ParseLineEntriesV1(entries)
    elif version == 'V2':
        meta_pos, meta_neg, meta_drop = ParseLineEntriesV2(entries)
    else:
        print('Invalid csv version: {}'.format(version))
        return

    print('pos: {}, neg: {}, drop: {}'.format(len(meta_pos), len(meta_neg), len(meta_drop)))

    save(meta_pos, output_prefix+'_pos.txt')
    save(meta_neg, output_prefix+'_neg.txt')
    save(meta_drop, output_prefix+'_drop.txt')


def parse_args():
    parser = argparse.ArgumentParser(description='Parse csv data to get fid and annotations')
    parser.add_argument('--csv', required=True,
                        help='CSV file to parse')
    parser.add_argument('--output-prefix', required=True,
                        help='Output files prefix, output files will be [output_prefix]_[train, test]_[pos, neg, drop].txt')
    parser.add_argument('--version', required=True,
                        help='Entries versions [V1, V2]')
    parser.add_argument('--train-ratio', type=int,
                        help='Split entries set into training set and test set according to this ratio',
                        default=0.9)
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the list before split')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    entries = open(args.csv).read().splitlines()
    num_entries = len(entries)
    num_train = int(num_entries * args.train_ratio)

    if args.shuffle:
        random.shuffle(entries)

    parse_file(entries[:num_train], args.output_prefix+'_train', args.version)
    if args.train_ratio < 1:
        parse_file(entries[num_train:], args.output_prefix+'_test', args.version)
