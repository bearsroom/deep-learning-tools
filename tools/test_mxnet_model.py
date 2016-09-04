
import sys
import os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(cwd, '..'))
import frameworks.find_mxnet
import mxnet as mx
import logging
import numpy as np
import argparse
import time
import logging
from multiprocessing import Process, Queue, Lock
from data_loaders.data_loader import BatchLoader
from predictors.predict_worker import predict_worker
import signal


processes = []
pid = None
NUM_PREPROCESSOR = 4


def signal_handler(signum, frame):
    print('Received exit signal...')
    if pid != os.getpid():
        return
    for p in processes:
        os.system('kill -9 {}'.format(p.pid))
        print('{} killed'.format(p.pid))
    sys.exit()


def random_pick(im_list, num_images):
    out_list = []
    ratio = num_images / float(len(im_list))
    for im in im_list:
        if np.random.rand() < ratio:
            out_list.append(im)
        if len(out_list) >= num_images:
            break
    return out_list


def test(im_list, data_dir, mean_img_file, model_params, output_prefix, classes, batch_size, gpus, data_shape=(3, 224, 224), evaluate=False):
    global processes
    assert len(im_list) > 0

    data_que_list = [Queue()]
    status_que = Queue()
    lock = Lock()
    # predict_worker
    logging.info('Initializing {} predictor...'.format(len(gpus)))
    for idx, gpu_id in enumerate(gpus):
        p = Process(target=predict_worker,
                    args=(idx, output_prefix+'_results.'+str(idx), classes, model_params, batch_size, data_que_list[0], lock, status_que),
                    kwargs={'gpu_id': gpu_id, 'evaluate': evaluate, 'framework': 'mxnet'})
        p.daemon = True
        p.start()
        processes.append(p)

    # make sure at leaset one predictor is available
    available = 0
    for _ in enumerate(gpus):
        status = status_que.get()
        if status == 'OK':
            available += 1
    if available == 0:
        logging.fatal('No predictor available! Exit...')
        for p in processes:
            p.join()
        return
    logging.info('Initialized {} predictors'.format(available))

    # create batch loader
    batch_loader = BatchLoader(im_list, data_dir, mean_img_file, NUM_PREPROCESSOR, data_que_list, data_shape=data_shape, batch_size=batch_size, gt_mode=evaluate, framework='mxnet')
    batch_loader.start_fetcher()
    loader_processes = batch_loader.get_processes()
    if len(loader_processes) == 0:
        raise ValueError('No loader processes! {}'.format(len(loader_processes)))
    processes += loader_processes
    logging.info('Start preprocessing images...')
    logging.info('Start predicting labels...')
    logging.info('Batch Loader starts providing batches...')
    batch_loader.provide_batch()

    for p in processes:
        p.join()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='GoogLeNet classifier: given a single image, return its label')
    parser.add_argument('--img-list', dest='im_list', required=True,
                        help='List of images to classify, if set, ignore --img flag',
                        default=None, type=str)
    parser.add_argument('--classes', dest='classes', required=True,
                        help='List of classes',
                        default=None, type=str)
    parser.add_argument('--data-dir', dest='data_dir', required=True,
                        help='Image data dir to join the paths',
                        default=None, type=str)
    parser.add_argument('--model-prefix', dest='model_prefix', required=True,
                        help='GoogLeNet model prefix',
                        default=None, type=str)
    parser.add_argument('--num-epoch', dest='num_epoch', required=True,
                        help='num of epoch to load',
                        default=None, type=int)
    parser.add_argument('--mean-img', dest='mean_img', required=True,
                        help='Mean image to substract',
                        default=None, type=str)
    parser.add_argument('--eval', dest='evaluate',
                        help='Mean image to substract',
                        action='store_true')
    parser.add_argument('--gpus', dest='gpus',
                        help="GPU devices to use, split by ','",
                        default='0', type=str)
    parser.add_argument('--output-prefix', dest='output_prefix', required=True,
                        help='Output file prefix, results will be stored in [output_prefix]_results.x where x a number',
                        default=None, type=str)
    parser.add_argument('--batch-size', help='Batch size', default=200, type=int)
    parser.add_argument('--num-images', default=None, type=int,
                        help='Number of images to test, if set, will randomly take num_images images to test')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global pid
    pid = os.getpid()
    args = parse_args()

    FORMAT = "%(asctime)s %(levelname)s %(process)d %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG, filename=None)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    im_list = open(args.im_list).read().splitlines()
    classes = open(args.classes).read().splitlines()
    # classes list format: int_label class_name class_thresh
    classes = [c.split()[1].strip() for c in classes]

    if args.num_images:
        logging.info('--num-images is set, will randomly pick {} images to test'.format(args.num_images))
        im_list = random_pick(im_list, args.num_images)

    logging.info('Test {} images, {} classes'.format(len(im_list), len(classes)))

    model_params = {'model_prefix': args.model_prefix, 'num_epoch': args.num_epoch}
    args.gpus = args.gpus.split(',')
    args.gpus = [int(g) for g in args.gpus]
    if len(args.gpus) > 1:
        if args.evaluate:
            logging.fatal('FATAL: currently evaluation mode does NOT support multi-gpu, exit')
            sys.exit(1)
    test(im_list, args.data_dir, args.mean_img, model_params, args.output_prefix, classes, args.batch_size, args.gpus, evaluate=args.evaluate)
