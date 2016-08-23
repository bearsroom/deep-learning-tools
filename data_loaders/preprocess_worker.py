
import logging
import numpy as np
from skimage import io, transform
import argparse
import sys
import os
import time
import logging


def preprocess_worker(proc_id, im_list, data_dir, mean_img_file, out_que, lock, data_shape=(3, 224, 224), gt_mode=False, framework='mxnet', crop_mode='center'):
    # load preprocessor corresponded to framework
    if framework.lower() == 'mxnet':
        from mxnet_preprocessor import MXNetPreprocessor
        preprocessor = MXNetPreprocessor(mean_img_file, data_shape=data_shape, crop_mode=crop_mode)
    elif framework.lower() == 'caffe':
        from caffe_preprocessor import CaffePreprocessor
        preprocessor = CaffePreprocessor(mean_img_file, data_shape=data_shape, crop_mode=crop_mode)

    idx = 0
    num_field = len(im_list[0].split())
    start = time.time()
    for line in im_list:
        if gt_mode:
            im_name, gt_label = line.split()[:2]
            gt_label = int(gt_label)
        else:
            im_name = line.split()[0]
            gt_label = None
        try:
            # load image
            im_path = os.path.join(data_dir, im_name)
            img = io.imread(im_path)
            # sanity check
            if len(img.shape) != 3:
                logging.warning('{}: not in color mode, drop'.format(im_name))
                continue
            elif img.shape[-1] != 3:
                logging.warning('{}: color image but nor in RGB mode, drop'.format(im_name))
                continue
            img = preprocessor.preprocess(img)
            lock.acquire()
            out_que.put((im_name, img, gt_label))
            lock.release()
            idx += 1
            if idx % 1000 == 0 and idx != 0:
                elapsed = time.time() - start
                logging.info('Preprocessor #{} processed {} images, elapsed {}s'.format(proc_id, idx, elapsed))
        except Exception, e:
            logging.error('{}: {}'.format(im_name, e))

    lock.acquire()
    out_que.put('FINISH')
    lock.release()
    elapsed = time.time() - start
    logging.info('Preprocessor #{} finished, processed {} images, elapsed {}s'.format(proc_id, idx, elapsed))
    return


