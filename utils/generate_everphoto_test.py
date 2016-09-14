
import os
import sys
import argparse
import random
from parse_entries import ParseLineEntriesV2


def create_tag_dict(tag_list):
    tag_dict = {}
    for line in tag_list:
        tag_dict[line[1]] = int(line[0])
    print(tag_dict)
    return tag_dict


def generate_class_test_list(class_name, class_id, data_dir, mode='pos'):
    class_file = '%s_test_%s.txt' % (class_name, mode)
    class_file = os.path.join(data_dir, class_file)
    if os.path.isfile(class_file):
        class_list = open(class_file).read().splitlines()
        class_list = [[c.split()[-1], str(class_id), class_name] for c in class_list]
        print('Got {} images of class {}'.format(len(class_list), class_name))
        return class_list
    else:
        return None


def generate_test_list(class_dict, data_dir, mode='pos'):
    """ output format: (fid, label_id, label_name)  """
    test_list = []
    for class_name, class_id in class_dict.items():
        cls_test_list = generate_class_test_list(class_name, class_id, data_dir, mode=mode)
        if cls_test_list:
            test_list += cls_test_list
        else:
            print('Cannot load test data of class {}'.format(class_name))
    return test_list


def single2multi(meta_list):
    meta_dict = {}
    for m in meta_list:
        if m[0] not in meta_dict:
            meta_dict[m[0]] = [(m[1], m[2])]
        else:
            meta_dict[m[0]].append((m[1], m[2]))
    output_list = []
    for im_name, meta in meta_dict.items():
        ids = ' '.join([str(m[0]) for m in meta]) if len(meta) > 1 else str(meta[0][0])
        tags = ' '.join([m[1] for m in meta]) if len(meta) > 1 else meta[0][1]
        output_list.append([im_name, ids, tags])
    return output_list


def add_other(meta_list, other_list):
    im_name_list = [m[0] for m in meta_list]
    for o in other_list:
        if o not in im_name_list:
            meta_list.append([o, str(-1), 'other'])
    return meta_list


def parse_args():
    parser = argparse.ArgumentParser(description='Generate test data for Everphoto DB, input lists format: _,..,fid')
    parser.add_argument('--classes', required=True,
                        help='Classes list with format (class_id, class_name, _,...)')
    parser.add_argument('--meta-dir', required=True,
                        help='Dir where store the test lists with format (_,...,fid)')
    parser.add_argument('--output', required=True,
                        help='Output list file')
    parser.add_argument('--mode', default='pos', choices=['pos', 'neg', 'drop'],
                        help='Mode of test list, valid choices: pos, neg, drop')
    parser.add_argument('--other',default=None,
                        help='Other images to add in test list, will be labeled as -1')
    parser.add_argument('--multilabel', action='store_true',
                        help='Allow multi-labels')
    parser.add_argument('--no-id', action='store_true',
                        help='do NOT output class id in list')
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    class_list = open(args.classes).read().splitlines()
    class_list = [c.split() for c in class_list]
    class_dict = create_tag_dict(class_list)

    test_list = generate_test_list(class_dict, args.meta_dir, mode=args.mode)
    if args.multilabel:
        test_list = single2multi(test_list)

    if args.other:
        other_list = set([o.split()[0].split('/')[-1] for o in open(args.other).read().splitlines()])
        test_list = add_other(test_list, other_list)

    random.shuffle(test_list)

    if args.no_id:
        test_list = [[t[0], t[-1]] for t in test_list]

    with open(args.output, 'w') as f:
        for t in test_list:
            f.write('{}\n'.format(' '.join(t)))
