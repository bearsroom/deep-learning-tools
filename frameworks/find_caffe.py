try:
    import caffe
except ImportError:
    import os, sys
    path = '/home/liyinghong/caffe-master/python'
    if not os.path.isdir(os.path.join(path, 'caffe')):
        raise ImportError

    sys.path.append(path)
    import caffe
