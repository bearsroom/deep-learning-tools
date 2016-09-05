
import os
import re
import sys
import argparse

def create_symbol_link(src_dir, images, dst_root, classes, limit=1000, suffix=''):
    print('Destination: {}'.format(dst_root))
    if not os.path.isdir(dst_root):
        os.mkdir(dst_root)
    for cls, im_cls in zip(classes, images):
        if im_cls == []:
            continue
        dst_dir = os.path.join(dst_root, cls)
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        print('Class {} dst dir: {}'.format(cls, dst_dir))
        for im in im_cls[:limit]:
        #for im in im_cls[-limit:]:
            src_dir = os.path.abspath(src_dir)
            src = os.path.join(src_dir, im)
            dst = os.path.join(dst_dir, os.path.basename(im))
            if suffix != '' and not dst.endswith(suffix):
                dst = dst + '.' + suffix
            if os.path.isfile(dst):
                print('dest {} exist'.format(dst))
                continue
            print('link {} to {}'.format(src, dst))
            os.symlink(src, dst)


def parse_images_file(image_file, classes):
    """ input format: (im_name label, ...) """
    images_list = open(image_file).read().splitlines()
    images_list = [im.split()[:2] for im in images_list]
    #if len(images_list[0]) == 5:
    #    images_list = [[im[0], im[2].rstrip(',')] for im in images_list]
    print('Will create {} symbol links'.format(len(images_list)))

    images = [[] for _ in range(len(classes))]
    for im in images_list:
        idx = classes.index(im[1])
        images[idx].append(im[0])

    return images


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Give a list of images with label, create symbol links for these images')
    parser.add_argument('--img-list', dest='im_list', required=True,
                        help='List of images with label to generate symbol links',
                        default=None, type=str)
    parser.add_argument('--classes', dest='classes', required=True,
                        help='List of classes',
                        default=None, type=str)
    parser.add_argument('--src-dir', dest='src_dir', required=True,
                        help='Image data source dir to join the paths',
                        default=None, type=str)
    parser.add_argument('--dst-dir', dest='dst_dir', required=True,
                        help='Destination root dir to store symbolink',
                        default=None, type=str)
    parser.add_argument('--limit', dest='limit',
                        help='Limit numbers of symbol links per class',
                        default=1000, type=int)
    parser.add_argument('--suffix', dest='suffix',
                        help='Suffix of symbol links, ex: jpg, png',
                        default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # classes format: (_, class_name, ...)
    classes = open(args.classes).read().splitlines()
    classes = [c.split()[1] for c in classes]
    print(classes)
    images = parse_images_file(args.im_list, classes)
    create_symbol_link(args.src_dir, images, args.dst_dir, classes, args.limit, args.suffix)
