
import find_caffe
import caffe
import mu


class CaffePreprocessor:
    def __init__(self, mean_npy, crop_mode='center', data_shape=(3, 224, 224)):
        super(CaffePreprocessor, self).__init__()
        self._init_transformer(mean_npy, data_shape)
        self.crop_mode = crop_mode

    def init_transformer(mean_npy, data_shape):
        mu = np.load(mean_npy)
        self.transformer = caffe.io.Transformer({'data': data_shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', mu)
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))

    def crop_image(self, img):
        # crop image
        if self.crop_mode == 'random':
            short_edge = min(img.shape[:2])
            yy = max(np.random.randint(img.shape[0] - short_edge + 1) - 1, 0)
            xx = max(np.random.randint(img.shape[1] - short_edge + 1) - 1, 0)
            crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
        else:
            if self.crop_mode != 'center':
                logging.warning('Currently we provide only random crop and center crop, use center crop by default')
            short_edge = min(img.shape[:2])
            yy = int((img.shape[0] - short_edge) / 2)
            xx = int((img.shape[1] - short_edge) / 2)
            img = img[yy:yy+short_edge, xx:xx+short_edge]
            return img

    def preprocess(self, img):
        img = self.crop_image(img)
        img = self.transformer.preprocess('data', img)
        return img


