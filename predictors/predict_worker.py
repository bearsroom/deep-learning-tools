
import logging
import os, sys
import time
import numpy as np


def load_model(proc_id, model_params, batch_size, classes, gpu_id, framework='mxnet'):
    """ model_params: dict, keys = ('model_prefix', 'num_epoch') if load mxnet model
        keys = ('prototxt', 'caffemodel') if load caffe model"""

    if framework.lower() == 'mxnet':
        from .mxnet_predictor import MXNetPredictor
        try:
            predictor = MXNetPredictor(model_params['model_prefix'], model_params['num_epoch'], batch_size, classes, gpu_id)
            # warm up with a dummy batch
            #logging.info('Warm up')
            #dummy = np.random.random((batch_size, 3, 224, 224))
            #prob = predictor.predict(dummy)
            #logging.info('a dummy batch went through')
            #if not isinstance(prob, np.ndarray):
            #    raise ValueError('Error when loading model')
            return predictor
        except (KeyError, ValueError) as e:
            logging.error('Predictor #{} failed to load mxnet model: {}'.format(proc_id, e))
            return None

    elif framework.lower() == 'caffe':
        from .caffe_predictor import CaffePredictor
        try:
            predictor = CaffePredictor(model_params['prototxt'], model_params['caffemodel'], batch_size, classes, gpu_id)
            return predictor
        except KeyError, e:
            logging.error('Predictor #{} failed to load caffe model: {}'.format(proc_id, e))
            return None


def predict_worker(proc_id, output_file, classes, model_params, batch_size, que, lock, status_que, gpu_id=0, evaluate=True, framework='mxnet'):
    """ get data from batch loader and make predictions, predictions will be saved in output_file
        if evaluate, will evaluate recall, precision, f1_score and recall_top5 """

    logging.info('Predictor #{}: Loading model...'.format(proc_id))
    model = load_model(proc_id, model_params, batch_size, classes, gpu_id, framework=framework)
    if model is None:
        status_que.put('Error')
        raise ValueError('No model created! Exit')
    logging.info('Predictor #{}: Model loaded'.format(proc_id))
    status_que.put('OK')

    if evaluate:
        from metrics import F1, ConfusionMatrix, MisClassified, RecallTopK
        evaluator = F1(len(classes))
        misclassified = MisClassified(len(classes))
        cm = ConfusionMatrix(classes)
        recall_topk = RecallTopK(len(classes), top_k=5)

    f = open(output_file, 'w')
    batch_idx = 0
    logging.info('Predictor #{} starts'.format(proc_id))
    start = time.time()
    while True:
        # get a batch from data loader via a queue
        lock.acquire()
        batch = que.get()
        lock.release()
        if batch == 'FINISH':
            logging.info('Predictor #{} has received all batches, exit'.format(proc_id))
            break

        # predict
        im_names, batch, gt_list = batch
        logging.debug('Predictor #{}: predict'.format(proc_id))
        pred, prob = model.predict(batch)
        pred_labels, top_probs = model.get_label_prob(top_k=5)

        # write prediction to file
        for im_name, label, top_prob in zip(im_names, pred_labels, top_probs):
            if im_name is None:
                continue
            top_prob = [str(p) for p in top_prob]
            f.write('{} labels:{} prob:{}\n'.format(im_name, ','.join(label), ','.join(top_prob)))

        # update metrics if evaluation mode is set
        if evaluate:
            assert gt_list is not None and gt_list != [] and gt_list[0] is not None
            top1_int = [p[0] for p in pred]
            assert len(top1_int) == len(gt_list), '{} != {}'.format(len(top1_int), len(gt_list))
            evaluator.update(top1_int, gt_list)
            misclassified.update(top1_int, gt_list, prob, im_names)
            cm.update(top1_int, gt_list)

            top5_int = [p[:5] for p in pred]
            assert len(top5_int) == len(gt_list), '{} != {}'.format(len(top5_int), len(gt_list))
            recall_topk.update(top5_int, gt_list)

        batch_idx += 1
        if batch_idx % 50 == 0 and batch_idx != 0:
            elapsed = time.time() - start
            logging.info('Predictor #{}: Tested {} batches of {} images, elapsed {}s'.format(proc_id, batch_idx, batch_size, elapsed))


    # evaluation after prediction if set
    if evaluate:
        logging.info('Evaluating...')
        recall, precision, f1_score = evaluator.get()
        for rec, prec, f1, cls, in zip(recall, precision, f1_score, classes):
            print('Class {:<20}: recall: {:<12}, precsion: {:<12}, f1 score: {:<12}'.format(cls, rec, prec, f1))
            f.write('Class {:<20}: recall: {:<12}, precsion: {:<12}, f1 score: {:<12}\n'.format(cls, rec, prec, f1))
        topk_recall = recall_topk.get()
        for rec, cls in zip(recall, classes):
            print('Class {:<20}: recall-top-5: {:<12}'.format(cls, rec))
            f.write('Class {:<20}: recall-top-5: {:<12}\n'.format(cls, rec))

        fp_images, fn_images = misclassified.get()
        g = open(output_file+'.fp', 'w')
        for cls, fp_cls in zip(classes, fp_images):
            for fp in fp_cls:
                g.write('{} pred:{} prob:{} gt:{} prob:{}\n'.format(fp[0], cls, fp[2], classes[fp[1]], fp[3]))
        g.close()
        g = open(output_file+'.fn', 'w')
        for cls, fn_cls in zip(classes, fn_images):
            for fn in fn_cls:
                g.write('{} gt:{} prob:{} pred:{} prob:{}\n'.format(fp[0], cls, fp[3], classes[fp[1]], fp[2]))
        g.close()

        cm.normalize()
        plt_name = output_file+'_cm.jpg'
        cm.draw(plt_name)
    f.close()


