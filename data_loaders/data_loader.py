
import logging
import numpy as np
import argparse
import sys
import os
import time
import logging
from multiprocessing import Process, Queue, Lock
from preprocess_worker import preprocess_worker

processes = []
LOAD_CAPACITY = 2000


class BatchLoader():
    def __init__(self, im_list, data_dir, mean_img_file, num_preprocessor, out_que_list, data_shape=(3, 224, 224), batch_size=100, gt_mode=False, framework='mxnet', crop_mode='center'):
        self.num_preprocessor = num_preprocessor
        self.batch_size = batch_size
        self.out_que_list = out_que_list
        # distribute im_list
        im_proc_list = [[] for _ in range(num_preprocessor)]
        for idx, line in enumerate(im_list):
            which_idx = idx % num_preprocessor
            im_proc_list[which_idx].append(line)

        self.processes = []
        self.que = Queue(LOAD_CAPACITY) # at most LOAD_CAPACITY images in queue
        self.lock = Lock()
        for idx in range(num_preprocessor):
            p = Process(target=preprocess_worker,
                        args=(idx, im_proc_list[idx], data_dir, mean_img_file, self.que, self.lock, gt_mode),
                        kwargs=(dict(gt_mode=gt_mode, data_shape=data_shape, framework=framework, crop_mode=crop_mode)))
            p.daemon = True
            self.processes.append(p)

    def start_fetcher(self):
        for p in self.processes:
            p.start()

    def provide_batch(self):
        batch_idx = 0
        while True:
            try:
                batch = self.get()
            except MemoryError, e:
                logging.info('BatchLoader: {}'.format(e))
                for que in self.out_que_list:
                    que.put('FINISH')
                logging.error('BatchLoader will exit due to unexpected MemoryError :(')
                break
            if batch:
                for que in self.out_que_list:
                    que.put(batch)
                    batch_idx +=1
                    if batch_idx % 10 == 0 and batch_idx != 0:
                        logging.info('BatchLoader send {} batches of {} images'.format(batch_idx, self.batch_size))
            else:
                for que in self.out_que_list:
                    que.put('FINISH')
                logging.info('BatchLoader exit...')
                break

    def get_processes(self):
        if len(self.processes) == 0:
            raise ValueError('No internal processes!')
        return self.processes

    def get(self):
        batch = np.zeros((self.batch_size, 3, 224, 224), dtype=np.float)
        im_names = [None] * self.batch_size
        gt = [None] * self.batch_size
        if self.num_preprocessor > 0:
            idx = 0
            while idx < self.batch_size:
                data = self.que.get()
                if data == 'FINISH':
                    self.num_preprocessor -= 1
                    if self.num_preprocessor <= 0:
                        logging.info('All preprocessors terminated')
                        break
                    else:
                        continue
                im_names[idx] = data[0]
                batch[idx] = data[1]
                gt[idx] = data[2]
                idx += 1
            return (im_names, batch, gt)
        else:
            logging.warning('No batch left')
            return None

    def __del__(self):
        for p in self.processes:
            p.join()
        logging.info('BatchLoader terminated')


