
from metrics import ConfusionMatrix
from metrics import Metrics
import find_caffe
import caffe
#import mxnet as mx
import logging
import numpy as np
#from skimage import io, transform
import argparse
import sys
import os
import time
import logging
from multiprocessing import Process, Queue, Lock
from data_loader import BatchLoader
import signal


processes = []
pid = None
BATCH_SIZE = 200
NUM_PREPROCESSOR = 4


def signal_handler(signum, frame):
    print('Received exit signal...')
    if pid != os.getpid():
        return
    for p in processes:
        os.system('kill -9 {}'.format(p.pid))
        print('{} killed'.format(p.pid))
    sys.exit()


def load_caffe_model(prototxt, caffemodel, batch_size, gpu_id=0):
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    try:
        model = caffe.Net(prototxt, caffemodel, caffe.TEST)
        old_batch_size = model.blobs['data'].data.shape[0]
        if old_batch_size != batch_size:
            old_shape = model.blobs['data'].data.shape
            new_shape = (batch_size, old_shape[1], old_shape[2], old_shape[3])
            model.blobs['data'].reshape(*new_shape)
    except Exception, e:
        logging.error('Error when loading model: {}'.format(e))
        return None
    return model


def predict(model, batch):
    assert model.blobs['data'].data.shape == batch.shape
    model.blobs['data'].data[...] = batch
    prob = model.forward()['prob']
    pred = np.argsort(prob, axis=1)[:,::-1]
    return pred, prob


def get_label_prob(pred, prob, classes, top_k=1):
    if top_k == 1:
        labels = [classes[p[0]] for p in pred]
        probs = np.sort(prob, axis=1)[:,-1]
        return labels, probs
    try:
        labels = [[classes[p[i]] for i in range(top_k)] for p in pred]
        probs = np.sort(prob, axis=1)[:,::-1][:,:top_k]
        return labels, probs
    except Exception, e:
        logging.error('Can\'t access class labels: {}'.format(e))
        return None, None


def predict_worker(proc_id, output_file, classes, prototxt, caffemodel, batch_size, in_que, lock, gpu_id=0, evaluate=True):
    logging.info('Predictor #{}: Loading model...'.format(proc_id))
    model = load_caffe_model(prototxt, caffemodel, batch_size, gpu_id)
    if model is None:
        raise ValueError('No model created! Exit')
    logging.info('Model loaded')

    f = open(output_file, 'w')
    if evaluate:
        evaluator = Metrics(len(classes))
        cm = ConfusionMatrix(classes)
    batch_idx = 0
    start = time.time()
    while True:
        try:
            lock.acquire()
            batch = in_que.get()
            lock.release()
        except MemoryError, e:
            logging.error('Predictor #{}: MemoryError: {}, skip'.format(proc_id))
            continue
        if batch == 'FINISH':
            logging.info('Predictor #{} has received all batches, exit'.format(proc_id))
            break

        im_names, batch, gt_list = batch
        pred, prob = predict(model, batch)
        pred_labels, top_probs = get_label_prob(pred, prob, classes, top_k=5)
        for im_name, label, top_prob in zip(im_names, pred_labels, top_probs):
            if im_name is None:
                continue
            top_prob = [str(p) for p in top_prob]
            f.write('{} labels:{} prob:{}\n'.format(im_name, ','.join(label), ','.join(top_prob)))
        batch_idx += 1
        if batch_idx % 50 == 0 and batch_idx != 0:
            elapsed = time.time() - start
            logging.info('Predictor #{}: Tested {} batches, elapsed {}s'.format(proc_id, batch_idx, elapsed))

        if evaluate:
            assert gt_list is not None and gt_list != [] and gt_list[0] is not None
            top1_int = [p[0] for p in pred]
            assert len(top1_int) == len(gt_list), '{} != {}'.format(len(top1_int), len(gt_list))
            evaluator.update_top1(top1_int, gt_list)
            evaluator.update_fp_images(top1_int, gt_list, prob, im_names)
            cm.update(top1_int, gt_list)

            top5_int = [p[:5] for p in pred]
            assert len(top5_int) == len(gt_list), '{} != {}'.format(len(top5_int), len(gt_list))
            evaluator.update_topk(top5_int, gt_list, top_k=5)

    if evaluate:
        logging.info('Evaluating...')
        recall, precision, f1_score = evaluator.get(metric='f1_score')
        for rec, prec, f1, cls, in zip(recall, precision, f1_score, classes):
            print('Class {:<20}: recall: {:<12}, precsion: {:<12}, f1 score: {:<12}'.format(cls, rec, prec, f1))
            f.write('Class {:<20}: recall: {:<12}, precsion: {:<12}, f1 score: {:<12}\n'.format(cls, rec, prec, f1))
        topk_recall = evaluator.get(metric='topk_recall')
        for rec, cls in zip(recall, classes):
            print('Class {:<20}: recall-top-5: {:<12}'.format(cls, rec))
            f.write('Class {:<20}: recall-top-5: {:<12}\n'.format(cls, rec))

        g = open(output_file+'.fp', 'w')
        fp_images = evaluator.get_fp_images()
        for cls, fp_cls in zip(classes, fp_images):
            for fp in fp_cls:
                g.write('{} pred:{} prob:{} gt:{} prob:{}\n'.format(fp[0], cls, fp[2], classes[fp[1]], fp[3]))
        g.close()

        cm.normalize()
        plt_name = output_file+'_cm.jpg'
        cm.draw(plt_name)

    f.close()


def test(im_list, data_dir, mean_img_file, model_list, output_prefix, classes, batch_size, gpus, evaluate=False):
    global processes
    assert len(im_list) > 0

    data_que_list = [Queue()]
    lock = Lock()
    # predict_worker
    logging.info('Initializing {} predictor...'.format(len(gpus)))
    for idx, gpu_id in enumerate(gpus):
        p = Process(target=predict_worker,
                    args=(idx, output_prefix+'_results.'+str(idx), classes, model_list[idx][0], model_list[idx][1], batch_size, data_que_list[0], lock),
                    kwargs={'gpu_id': gpu_id, 'evaluate': evaluate})
        p.daemon = True
        p.start()
        processes.append(p)

    # create batch loader
    #batch_loader = BatchLoader(im_list, data_dir, mean_img_file, NUM_PREPROCESSOR, data_que_list, batch_size=batch_size, gt_mode=evaluate, minus_mean_after_resize=False, color_mode='BGR')
    batch_loader = BatchLoader(im_list, data_dir, mean_img_file, NUM_PREPROCESSOR, data_que_list, batch_size=batch_size, gt_mode=evaluate, caffe_mode=True)
    batch_loader.start_fetcher()
    loader_processes = batch_loader.get_processes()
    logging.info('Loader has {} internal processes, type {}'.format(len(loader_processes), type(loader_processes[0])))
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
    parser.add_argument('--prototxt', dest='prototxt', required=True,
                        help='GoogLeNet model prefix',
                        default=None, type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', required=True,
                        help='num of epoch to load',
                        default=None, type=str)
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
    parser.add_argument('--data-shape', help='Data shape (height, width), height = width', default=224, type=int)

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
    logging.basicConfig(format=FORMAT, level=logging.INFO, filename=None)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    im_list = open(args.im_list).read().splitlines()
    classes = open(args.classes).read().splitlines()
    # classes list format: int_label class_name class_thresh
    classes = [c.split()[1].strip() for c in classes]
    logging.info('Test {} images, {} classes'.format(len(im_list), len(classes)))

    model_list = [(args.prototxt, args.caffemodel)]
    args.gpus = args.gpus.split(',')
    args.gpus = [int(g) for g in args.gpus]
    if len(args.gpus) > 1:
        if args.evaluate:
            logging.fatal('FATAL: currently evaluation mode does NOT support multi-gpu, exit')
            sys.exit(1)
        model_list = model_list * len(args.gpus)
    test(im_list, args.data_dir, args.mean_img, model_list, args.output_prefix, classes, args.batch_size, args.gpus, evaluate=args.evaluate)
