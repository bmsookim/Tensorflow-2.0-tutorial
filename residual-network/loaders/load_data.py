import yaml
import tensorflow as tf
from loaders.dataloader import DataLoader

def load_data(cfg):
    train_data = DataLoader(cfg, split='train')
    test_data = DataLoader(cfg, split='test')

    num_train = len(train_data)
    num_test = len(test_data)

    train_data = tf.data.Dataset.from_generator(train_data, (tf.float32, tf.int32))
    test_data = tf.data.Dataset.from_generator(test_data, (tf.float32, tf.int32))

    train_data = train_data.batch(batch_size=cfg['SOLVER']['BATCH_SIZE'], drop_remainder=False)
    test_data = test_data.batch(batch_size=cfg['SOLVER']['BATCH_SIZE'], drop_remainder=False)

    return train_data, test_data

if __name__ == "__main__":
    cfg = yaml.load(open('../configs/cifar10.yaml'), Loader=yaml.FullLoader)
    train_data, test_data = load_data(cfg)

    image_batch, label_batch = next(iter(train))
