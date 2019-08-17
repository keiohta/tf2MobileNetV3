import os
import json

import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from goselo_rl.experiments.trainer import Trainer
from goselo_rl.misc.dataset import load_dataset


def train():
    parser = Trainer.get_argument()
    args = parser.parse_args()

    save_dir = "results"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    x_train, y_train, x_test, y_test, output_dim = load_dataset(
        dataset_name=args.dataset, max_size=args.max_size,
        dataset_name_prefix=args.dataset_prefix,
        split_ratio=args.test_split_ratio, shuffle=True)

    # from model.mobilenet_v3_small import MobileNetV3_Small
    # model = MobileNetV3_Small(x_train.shape[1:], output_dim).build()
    from goselo_rl.networks.mobilenet_v3_small import  MobileNetV3Small
    model = MobileNetV3Small(x_train.shape[1:], output_dim).model   

    # pre_weights = cfg['weights']
    # if pre_weights and os.path.exists(pre_weights):
    #     model.load_weights(pre_weights, by_name=True)

    opt = Adam(lr=0.001)
    earlystop = EarlyStopping(
        monitor='val_acc', patience=5, verbose=0, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    model.fit(
        x=x_train, y=y_train, batch_size=32, epochs=100,
        verbose=1, callbacks=None, validation_split=0.0,
        validation_data=(x_test, y_test), shuffle=True)

    model.save_weights(os.path.join(
        save_dir, '{}_weights.h5'.format("MobileNetV3")))


if __name__ == '__main__':
    train()
