#!/bin/bash

python split_data_by_tag.py --tags ~/data/classify_200_mxnet/db_meta/landscape/landscape_classes.txt --im-list /temp-hdd/liyinghong/data/everphoto_media/subset_results/landscape/results_rnd_2M_20160823/hard_entropy/results_2M_furnishing.hard.entropy --output-prefix /temp-hdd/liyinghong/data/everphoto_media/subset_results/landscape/results_rnd_2M_20160823/hard_entropy/images --topk 2 --not-filter

