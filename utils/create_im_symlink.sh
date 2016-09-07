#!/bin/bash


#for line in $(ls /temp-hdd/liyinghong/data/everphoto_media/subset_results/transport/easy/images_*); do python create_im_symlink.py --src-dir /temp-hdd/liyinghong/data/everphoto_media/rnd_2M_20160823 --dst-dir /temp-hdd/liyinghong/data/everphoto_media/subset_results/transport/easy/images --classes classify_200_mxnet/db_meta/transport/transport_classes.txt --img-list $line --suffix jpg --limit 500; done

for line in $(ls everphoto_media/subset_results/animal/images_*); do python create_im_symlink.py --src-dir everphoto_media/rnd_2M --dst-dir everphoto_media/subset_results/animal/images --classes classify_200_mxnet/db_meta/animal/animal_classes.txt --img-list $line --suffix jpg --limit 500; done

#for line in $(ls everphoto_media/subset_results/furnishing/hard_entropy/images_*); do python create_im_symlink.py --src-dir everphoto_media/rnd_2M --dst-dir everphoto_media/subset_results/furnishing/hard_entropy/images --classes classify_200_mxnet/db_meta/furnishing/furnishing_classes.txt --img-list $line --suffix jpg --limit 300; done

#for line in $(ls /temp-hdd/liyinghong/data/everphoto_media/subset_results/furnishing/images_2M_20160823_*); do python create_im_symlink.py --src-dir /temp-hdd/liyinghong/data/everphoto_media/rnd_2M_20160823 --dst-dir /temp-hdd/liyinghong/data/everphoto_media/subset_results/furnishing/images --classes classify_200_mxnet/db_meta/furnishing/furnishing_classes.txt --img-list $line --suffix jpg --limit 500; done
