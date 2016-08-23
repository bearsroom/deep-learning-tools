try:
    import mxnet as mx
except ImportError:
    import os, sys
    path = '/home/liyinghong/mxnet/python'
    if not os.path.isdir(os.path.join(path, 'mxnet')):
        raise ImportError

    sys.path.append(path)
    import mxnet as mx
