import tensorflow as tf

def normalizer(x: tf.Tensor) -> tf.Tensor:
    return x

def horizontal_flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    return x

def zoom(x: tf.Tensor) -> tf.Tensor:
    return x

def augment_data(dataset):
    augmentations = [horizontal_flip, zoom]
    for f in augmentations:
        dataset = dataset.map(lambda image, label: (f(image), label))
    return dataset
