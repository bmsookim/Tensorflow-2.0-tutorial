import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.applications.resnet import ResNet50

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        zoom_range=0.5
        )

data_dir = '/data/cifar10/train'
CLASS_NAMES = os.listdir(data_dir)

train_data_gen = train_generator.flow_from_directory(
        directory = data_dir,
        batch_size = 2,
        shuffle = True,
        target_size = (224, 224),
        classes = CLASS_NAMES
)

IMG_SHAPE = (224, 224, 3)
base_model = tf.keras.applications.ResNet50(
    input_shape = IMG_SHAPE,
    include_top = True,
    weights = 'imagenet'
)

for idx, data in enumerate(train_data_gen):
    image_batch, label_batch = data
    print(image_batch.shape)
    print(label_batch.shape)

    pred = base_model(data)
    print(pred.shape)

    break
