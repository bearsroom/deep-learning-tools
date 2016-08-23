try:
    import mxnet as mx
except ImportError:
    import os, sys
    paths = [p.split()  for p in open('framework_path.txt').read().splitlines()]
    path = ''
    for p in paths:
        if p[0] == 'MXNET_PATH':
            path = p[1]
            break
    if not os.path.isdir(os.path.join(path, 'mxnet')):
        raise ImportError

    sys.path.append(path)
    import mxnet as mx
