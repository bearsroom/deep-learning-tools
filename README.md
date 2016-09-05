Deep-Learning-Tools

Some useful tools to automate training and testing procedures

usage:

- tools/test_caffe_models.py test caffemodel and evaluate performance if ground truth provided
- tools/test_mxnet_models.py test mxnet model and evaluate

- sampling/al_sampling.py sampling images for further annotations using some active learning method:
  - min max sampling
  - margin sampling
  - entropy sampling

- utils/split_data_by_tag.py split prediction result list to mutiple lists, 1 list per tag, can pick top k or filter by thresh lower bound or upper bound
- utils/create_symbol_link.py create symbol links of images

- annotation/generate_annotation_samples.py generate samples for annotation (easy and hard samples)
