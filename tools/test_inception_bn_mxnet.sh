#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: ./train_nin.sh subset_name"
    exit 1
fi

declare -A subset
subset[construction_num_classes]=13
subset[construction_num_samples]=247484
subset[landscape_num_classes]=18
subset[landscape_num_samples]=343283
subset[text_num_classes]=10
subset[text_num_samples]=190649
subset[food_num_classes]=17
subset[food_num_samples]=323637
subset[plant_num_classes]=6
subset[plant_num_samples]=115220
subset[root_num_classes]=18
subset[root_num_samples]=458538



subset_class=$1
num_classes=${subset["$1_num_classes"]}
num_samples=${subset["$1_num_samples"]}


python test_mxnet_model.py \
--data-dir=/ \
--img-list=/home/liyinghong/data/classify_200_mxnet/db_meta/"$subset_class"/"$subset_class"_test_list.txt \
--model-prefix=/home/liyinghong/data/classify_200_mxnet/inception_bn/inception_bn_20160821_"$subset_class" \
--num-epoch=25 \
--mean-img=/home/liyinghong/data/classify_200_mxnet/db_meta/"$subset_class"/test_mean.nd \
--gpus=3 \
--classes=/home/liyinghong/data/classify_200_mxnet/db_meta/"$subset_class"/"$subset_class"_classes.txt \
--output-prefix=inception_bn_20160823_test_log_"$subset_class" \
--eval \
#--img-list=/home/liyinghong/data/everphoto_media/dataset_meta/"$subset_class"_test_everphoto_list.txt.r \
#--data-dir=/home/liyinghong/data/everphoto_media/rnd_2M/ \

