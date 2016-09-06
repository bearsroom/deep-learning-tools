
import sys
import argparse
import math


def create_tag_dict(tag_list, thresh_bound=False):
    tag_dict = {}
    for line in tag_list:
        if line[1] == 'none':
            continue
        if len(line) >= 4 and thresh_bound:
            tag_dict[line[1]] = [float(line[2]), float(line[3])]
        else:
            tag_dict[line[1]] = float(line[2])
    print(tag_dict)
    return tag_dict


def split_by_tag(tag_dict, im_list, filter_thresh=True, mode=None, topk=None):
    im_tag_dict = {}
    for tag in tag_dict.keys():
        im_tag_dict[tag] = []

    for im in im_list:
        try:
            im_name = im[0]
            tags = im[1]
            probs = im[2]
            if mode in ('fp', 'fn'):
                gt_tag = tags[-1]
                tags = tags[:-1]
                gt_prob = probs[-1]
                probs = probs[:-1]
                if topk and type(topk) is int:
                    probs = probs[:topk]
                for tag, prob in zip(tags, probs):
                    if tag in tag_dict.keys():
                        key = tag if mode == 'fp' else gt_tag
                        if filter_thresh and prob > tag_dict[tag]:
                            im_tag_dict[key].append((im_name, tag, prob, gt_tag, gt_prob))
                        elif not filter_thresh:
                            im_tag_dict[key].append((im_name, tag, prob, gt_tag, gt_prob))
            else:
                if topk and type(topk) is int:
                    probs = probs[:topk]
                    tags = tags[:topk]
                for tag, prob in zip(tags, probs):
                    if tag in tag_dict.keys():
                        if filter_thresh:
                            if type(tag_dict[tag]) is list and prob > tag_dict[tag][0] and prob <= tag_dict[tag][1]:
                                im_tag_dict[tag].append((im_name, tag, prob))
                            elif prob > tag_dict[tag]:
                                im_tag_dict[tag].append((im_name, tag, prob))
                        else:
                            im_tag_dict[tag].append((im_name, tag, prob))
        except Exception, e:
            continue

    return im_tag_dict


def intersect_dict(im_tag_dict, intersect_list):
    for tag, im_list in im_tag_dict.items():
        images = [im[0] for im in im_list]
        print('Class {} has {} images before intersection'.format(tag, len(images)))
        intersect_list = list(set(images) & set(intersect_list))
        im_list = [im for im in im_list if im[0] in intersect_list]
        im_tag_dict[tag] = im_list

    return im_tag_dict


def parse_line(line, mode=None):
    try:
        data = line.split()
        if mode in ('fp', 'fn'):
            im_name = data[0]
            preds = data[1].split(':')[-1].split(',')
            pred_probs = data[2].split(':')[-1].split(',')
            gt = data[3].split(':')[-1]
            gt_prob = data[4].split(':')[-1]
            labels = preds + [gt]
            probs = pred_probs + [gt_prob]
        else:
            im_name = data[0]
            labels = data[1].split(':')[-1].split(',')
            probs = data[2].split(':')[-1].split(',')
        probs = [float(p) for p in probs]
        return (im_name, labels, probs)
    except Exception:
        return None


def main(im_list_file, tag_id_list_file, output_prefix, intersect=None, mode=None, filter_thresh=True, thresh_bound=False, topk=None):
    if mode == 'fp':
        print('Enter FP mode, will split fp results')
    elif mode == 'fn':
        print('Enter FN mode, will split fn results')
    im_list = open(im_list_file).read().splitlines()
    im_list = [parse_line(im, mode=mode) for im in im_list]
    num_images = len(im_list)
    print('Total {} images in dataset'.format(num_images))

    tag_id_list = open(tag_id_list_file).read().splitlines()
    tag_id_list = [tag.split() for tag in tag_id_list]

    tag_dict = create_tag_dict(tag_id_list, thresh_bound=thresh_bound)

    im_tag_dict = split_by_tag(tag_dict, im_list, mode=mode, filter_thresh=filter_thresh, topk=topk)
    if intersect:
        intersect_list = open(intersect).read().splitlines()
        intersect_list = [inter.split()[0] for inter in intersect_list]
        im_tag_dict = intersect_dict(im_tag_dict, intersect_list)

    predict = 0
    for tag, prob in tag_dict.items():
        tag_ = tag
        if '/' in tag:
            tag_ = tag.replace('/', '_')
        if mode == 'fp':
            with open(output_prefix+'_{}.txt.fp'.format(tag_), 'w') as f:
                for line in im_tag_dict[tag]:
                    f.write('{} {} {} {} {}\n'.format(line[0], line[1], line[2], line[3], line[4]))
        elif mode == 'fn':
            with open(output_prefix+'_{}.txt.fn'.format(tag_), 'w') as f:
                for line in im_tag_dict[tag]:
                    f.write('{} {} {} {} {}\n'.format(line[0], line[3], line[4], line[1], line[2]))
        else:
            with open(output_prefix+'_{}.txt'.format(tag_) ,'w') as f:
                for line in im_tag_dict[tag]:
                    f.write('{} {} {}\n'.format(line[0], line[1], line[2]))
        print('Class {} has {} images, threshold {}'.format(tag, len(im_tag_dict[tag]), prob))
        predict += len(im_tag_dict[tag])
    print('Total {}/{} images extracted'.format(predict, num_images))


def parse_args():
    parser = argparse.ArgumentParser(description='Split results list by tags')
    parser.add_argument('--tags', required=True,
                        help='Tag id list, in format tag_id, tag_name, tag_threshold')
    parser.add_argument('--im-list', required=True,
                        help='Image results list, in format im_name, tag_name, tag_prob')
    parser.add_argument('--output-prefix', required=True,
                        help='Output prefix, will output results to [output_prefix]_tag_name.txt')
    parser.add_argument('--fp-mode', action='store_true',
                        help='FP mode, if set, expected fp image list as input')
    parser.add_argument('--fn-mode', action='store_true',
                        help='FN mode, if set, expected fp image list as input')
    parser.add_argument('--intersect', default=None,
                        help='Image list to perform intersection with [im_list]')
    parser.add_argument('--thresh-bound', action='store_true',
                        help='Bound output images\' probs in intervals')
    parser.add_argument('--not-filter', action='store_true',
                        help='Not to filter with threshold')
    parser.add_argument('--topk', default=None, type=int, help='Only pick top k prediction')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    filter_thresh = not args.not_filter

    if args.fp_mode:
        main(args.im_list, args.tags, args.output_prefix, args.intersect, 'fp', filter_thresh, args.thresh_bound, args.topk)
    elif args.fn_mode:
        main(args.im_list, args.tags, args.output_prefix, args.intersect, 'fn', filter_thresh, args.thresh_bound, args.topk)
    else:
        main(args.im_list, args.tags, args.output_prefix, args.intersect, filter_thresh=filter_thresh, thresh_bound=args.thresh_bound, topk=args.topk)

