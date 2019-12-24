import tensorflow as tf
import numpy as np
import os
import yaml
import pathlib
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
#from tensorflow.python.keras.applications.resnet import ResNet50

class DataLoader:
    def __init__(self, cfg, split):
        self.data_dir = pathlib.Path(cfg['DATASET']['DATA_DIR'])
        self.list_ds = tf.data.Dataset.list_files(
            file_pattern="{}/{}/*/*".format(self.data_dir, split),
            shuffle=(split=='train'),
        )
        self.CLASS_NAMES = os.listdir(cfg['DATASET']['TRAIN_DIR'])
        self.w, self.h = cfg['DATASET']['IMG_SHAPE']
        self.labeled_ds = self.list_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.labeled_ds = self.labeled_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def get_label(self, file_path):
        # path -> class-directory
        parts = tf.strings.split(file_path, os.path.sep)[-2]
        return parts == self.CLASS_NAMES

    def decode_img(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        # put augmentation here
        return tf.image.resize(img, [self.w, self.h])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def __call__(self):
        return self.labeled_ds

    def __len__(self):
        _len, run_iter = 0, self.labeled_ds.repeat(1)
        for idx in iter(run_iter): _len += 1
        return _len

    """
    def __call__(self):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        augmentations = [flip, zoom, normalization]

        for aug in augmentations:
            dataset = dataset.map(lambda x: tf.cond(tf.random_uniform))
    """

###########################################################################
# Convenient, but slow, lacks fine-grained control, and not well integrated
def create_dataloaders(cfg):
    CLASS_NAMES = os.listdir(cfg['DATASET']['TRAIN_DIR'])
    w, h = cfg['DATASET']['IMG_SHAPE']

    # >>> augmenters
    train_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_epsilon=1e-6,
        zca_whitening=False,
        zoom_range=0,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=False,
    )

    test_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
        rotation_range=0,
        width_shift_range=.0,
        height_shift_range=.0,
        zoom_range=0
    )

    # >>> data generators
    train_generator = train_augmenter.flow_from_directory(
        directory = cfg['DATASET']['TRAIN_DIR'],
        target_size = (w, h),
        color_mode = 'rgb',
        classes = CLASS_NAMES,
        class_mode = 'categorical',
        batch_size = cfg['SOLVER']['BATCH_SIZE'],
        shuffle = True,
        seed = None,
        save_to_dir = None,
        save_prefix = '',
        save_format = 'png',
        follow_links = False,
        subset = None,
        interpolation = 'nearest'
    )

    test_generator = test_augmenter.flow_from_directory(
        directory = cfg['DATASET']['TEST_DIR'],
        target_size = (w, h),
        color_mode = 'rgb',
        classes = CLASS_NAMES,
        class_mode = 'categorical',
        batch_size = cfg['SOLVER']['BATCH_SIZE'],
        shuffle = False,
        seed = None,
        save_to_dir = None,
        save_prefix = '',
        save_format = 'png',
        follow_links = False,
        subset = None,
        interpolation = 'nearest')

    return train_generator, test_generator, CLASS_NAMES
#####################################################################

if __name__ == '__main__':
    cfg = yaml.load(open('./configs/cifar10.yaml'), Loader=yaml.FullLoader)
    #train_generator, test_generator, class_names = create_dataloaders(cfg)

    #for idx, data in enumerate(train_generator):
    #    image_batch, label_batch = data
    #    print(image_batch.shape)
    #    print(label_batch.shape)
    #    break
    dataloader = DataLoader(cfg, 'train')
