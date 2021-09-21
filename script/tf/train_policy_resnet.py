import logging
from pathlib import Path
import sys
import os
import pickle
import random
import argparse
from tensorflow.keras.layers import Permute, Activation, Flatten, Conv2D, Layer, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers, Input, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from pydlshogi.read_kifu import *
from pydlshogi.features import *
from pydlshogi.time_log import TimeLog
from pydlshogi.common import *
import numpy as np

import pydlshogi
import pydlshogi.util


class BiasLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)

    def call(self, x):
        return x + self.bias


def residualBlock(input):
    ch = 192
    x = Conv2D(ch, kernel_size=(3, 3), use_bias=False, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(ch, kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = layers.add([input, x])
    x = Activation('relu')(x)
    return x


def createModel(blocks):
    ch = 192
    inputs = Input(shape=(9, 9, 104), name="digits")
    x = Conv2D(ch, kernel_size=(3, 3),
               activation='relu', padding='same')(inputs)
    for i in range(blocks):
        x = residualBlock(x)
    x = Conv2D(27, kernel_size=(1, 1),
               activation='relu', use_bias=False)(x)
    x = Permute((3, 1, 2))(x)
    x = Flatten()(x)
    x = BiasLayer()(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_argument_for_train(parser)
    parser.add_argument('--blocks', type=int, default=5,
                        help='Number of resnet blocks')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of positions in each mini-batch')
    parser.add_argument('--test_batchsize', type=int, default=512,
                        help='Number of positions in each test mini-batch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=1, help='Number of epoch times')
    parser.add_argument('--model', help='model dir name')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--log', default=None, help='log file path')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--eval_interval', '-i', type=int,
                        default=1000, help='eval interval')
    parser.add_argument('--train_loop', action='store_true')
    args = parser.parse_args()
    epochs = args.epoch
    batchsize = args.batchsize
    lr = args.lr
    eval_interval = args.eval_interval
    initmodel = args.initmodel
    save_model_dir = args.model

    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

    logger = logging.getLogger()
    logger.info('read kifu start')
    # 訓練用とテスト用の棋譜リストを取得
    time_load_position_list = TimeLog("load_position_list")
    time_load_position_list.start()
    df_positionlist_train, df_positionlist_test = load_positionlist(args)
    time_load_position_list.end()
    logger.info(f'train:{df_positionlist_train.describe()}')
    logger.info(f'test:{df_positionlist_test.describe()}')

    columns = [c for c in df_positionlist_train.columns if c[:2]
               == 'F_' and c != 'F_current_movenum']
    single_board_list_train = df_positionlist_train.loc[:, columns].values
    move_train = df_positionlist_train['L_hand'].values
    # win_train = df_positionlist_train['L_win'].values

    single_board_list_test = df_positionlist_test.loc[:, columns].values
    move_test = df_positionlist_test['L_hand'].values
    # win_test = df_positionlist_test['L_win'].values

    train_num = len(single_board_list_train)
    test_num = len(single_board_list_test)
    logger.info('train position num = {}'.format(train_num))
    logger.info('test position num = {}'.format(test_num))

    # train
    time_mini_batch = TimeLog("mini_batch")
    time_train = TimeLog("train")
    time_val_mini_batch = TimeLog("val_mini_batch")
    time_val_epoch = TimeLog("val_epoch")

    logger.info('start training')

    model = createModel(args.blocks)
    model_summary = []
    model.summary(
        print_fn=lambda x: model_summary.append(x))
    logger.info("\n".join(model_summary))

    # Init/Resume
    if initmodel:
        logging.info('Load model from {}'.format(args.initmodel))
        load_model("initmodel")

    # train
    logger.info('start training')

    # Instantiate an optimizer.
    optimizer = SGD(learning_rate=lr)

    # Instantiate a loss function.
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (single_board_list_train, move_train))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (single_board_list_test, move_test))

    train_dataset = train_dataset.shuffle(len(train_dataset)).batch(32).map(lambda x, y:
                                                                            (tf.numpy_function(func=make_input_features_from_single_board_list, inp=[x], Tout=tf.int8), y))
    test_dataset = test_dataset.batch(32).map(lambda x, y:
                                              (tf.numpy_function(func=make_input_features_from_single_board_list, inp=[x], Tout=tf.int8), y))

    logger.info(f'x_train num = {train_num}')
    logger.info(f'x_test num = {test_num}')

    time_train.start()
    # チェックポイントコールバックを作る
    callbacks = []
    if save_model_dir:
        callbacks.append(ModelCheckpoint(os.path.join(save_model_dir, 'weights-{epoch:04d}-{val_loss:.2f}.ckpt'),
                                         save_weights_only=True,
                                         verbose=1))

    model.compile(optimizer=optimizer, loss=loss_fn,
                  metrics=['accuracy'])
    history = model.fit(train_dataset, epochs=epochs, batch_size=batchsize,
                        validation_data=test_dataset, callbacks=callbacks, verbose=1)
    time_train.end()

    logging.info('save the model')
    if save_model_dir:
        model.save(os.path.join(save_model_dir, 'model'))
        pydlshogi.util.save_history(
            history, os.path.join(save_model_dir, 'history.pickle'))
        pydlshogi.util.save_history(
            history, os.path.join(save_model_dir, 'history.csv'))

    logger.info(f'\n{TimeLog.debug()}')
