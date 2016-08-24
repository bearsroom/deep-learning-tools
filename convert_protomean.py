
import frameworks.find_caffe
import caffe
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        print('Usage: python convert_protomean.py mean_file.binaryproto out.npy [resize=224]')
        sys.exit()

    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(sys.argv[1], 'rb').read()
    blob.ParseFromString(data)

    arr = np.array(caffe.io.blobproto_to_array(blob))
    arr = arr[0]

    if len(sys.argv) == 4:
        resize_shape = int(sys.argv[3].split('=')[-1])
        if min(arr.shape[1:]) < resize_shape:
            print('Error: cannot resize mean img to a larger size: {} to ({}, {})'.format(arr.shape[1:], resize_shape, resize_shape))
            sys.exit()
        arr = arr[:,:resize_shape,:resize_shape]

    np.save(sys.argv[2], arr)
