#!/bin/bash


subset=landscape

python roc_pr_curve_from_list.py \
--tags=/home/liyinghong/data/classify_200_mxnet/db_meta/"$subset"/"$subset"_classes.txt \
--im-list=/home/liyinghong/data/everphoto_media/dataset_meta/"$subset"_test/"$subset"_20160830_with_neg_results.0 \
--gt-list=/home/liyinghong/data/everphoto_media/dataset_meta/"$subset"_test/"$subset"_test_list.txt \
--output-prefix=/home/liyinghong/data/everphoto_media/dataset_meta/"$subset"_test/"$subset" \
--title="$subset" \
--min-precision=0.8 \
