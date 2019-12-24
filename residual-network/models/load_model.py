import tensorflow as tf
from models.ResNet import resnet

def load_model(cfg):
    model = resnet(18)
    sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
