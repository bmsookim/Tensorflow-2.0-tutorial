import tensorflow as tf
import sys
import os
import yaml
from loaders.load_data import load_data
from loaders.augmentation import augment_data

def main():
    cfg = yaml.load(open('./configs/cifar10.yaml'), Loader=yaml.FullLoader)
    train_data, test_data = load_data(cfg)

    for epoch in range(1, cfg['SOLVER']['EPOCHS']):
        #train_data = augment_data(train_data)
        for idx, (tr_batches, tr_labels) in enumerate(iter(train_data)):
            print(tr_batches)
            #print(tr_batches)

if __name__ == "__main__":
    main()
