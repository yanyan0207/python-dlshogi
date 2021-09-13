import numpy as np

from pydlshogi.common import *
from pydlshogi.time_log import TimeLog
from pydlshogi.features import *
from pydlshogi.read_kifu import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.python.keras.layers.core import Permute

import argparse
import random
import pickle
import os
import sys
from pathlib import Path

from hurry.filesize import size
import logging
from memory_profiler import profile


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)

    def call(self, x):
        return x + self.bias


def createModel(blocks):
    ch = 192
    inputs = keras.Input(shape=(9, 9, 104), name="digits")
    x = Conv2D(ch, kernel_size=(3, 3),
               activation='relu', padding='same')(inputs)
    for i in range(blocks):
        x = Conv2D(ch, kernel_size=(3, 3),
                   activation='relu', padding='same')(x)
    x = Conv2D(27, kernel_size=(1, 1),
               activation='relu', use_bias=False)(x)
    x = Permute((2, 3, 1))(x)
    x = Flatten()(x)
    x = BiasLayer()(x)
    outputs = x
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_argument_for_train(parser)
    parser.add_argument('--kifulist_root')
    parser.add_argument('--blocks', type=int, default=11,
                        help='Number of resnet blocks')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of positions in each mini-batch')
    parser.add_argument('--test_batchsize', type=int, default=512,
                        help='Number of positions in each test mini-batch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=1, help='Number of epoch times')
    parser.add_argument(
        '--model', type=str, default='model/model_policy_tf', help='model file name')
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
    save_model = args.model

    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

    logger = logging.getLogger()
    logger.info('read kifu start')
    # 訓練用とテスト用の棋譜リストを取得
    df_positionlist_train, df_positionlist_test = load_positionlist(args)
    logger.info(f'train:{df_positionlist_train.describe()}')
    logger.info(f'test:{df_positionlist_test.describe()}')

    columns = [c for c in df_positionlist_train.columns if c[:2]
               == 'F_' and c != 'F_current_movenum']
    single_board_list_train = df_positionlist_train.loc[:, columns].values
    move_train = df_positionlist_train['L_hand'].values
    #win_train = df_positionlist_train['L_win'].values

    single_board_list_test = df_positionlist_test.loc[:, columns].values
    move_test = df_positionlist_test['L_hand'].values
    #win_test = df_positionlist_test['L_win'].values

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
        keras.models.load_model("initmodel")

    # train
    logger.info('start training')

    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=lr)

    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if not args.train_loop:
        time_mini_batch.start()
        x_train = make_input_features_from_single_board_list(
            single_board_list_train)
        t1_train = move_train
        time_mini_batch.end()
        logger.info(f'x_train num = {train_num}')
        logger.info(f'x_train bytessize = {size(x_train.nbytes)}')

        time_val_mini_batch.start()
        x_test = make_input_features_from_single_board_list(
            single_board_list_test)
        t1_test = move_test
        time_val_mini_batch.end()
        logger.info(f'x_test num = {test_num}')
        logger.info(f'x_test bytessize = {size(x_test.nbytes)}')

        time_train.start()
        model.compile(optimizer=optimizer, loss=loss_fn,
                      metrics=['accuracy'])
        model.fit(x_train, t1_train, epochs=epochs,
                  batch_size=batchsize, validation_data=(x_test, t1_test), verbose=1)
        time_train.end()

        logging.info('save the model')
        model.save(save_model)

    else:
        # Prepare the metrics.
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        for e in range(epochs):
            logger.info("\nStart of epoch %d" % (e+1,))

            # Todo shuffle

            sum_loss = []
            for step, i in enumerate(range(0, train_num - batchsize, batchsize)):
                time_mini_batch.start()
                x = make_input_features_from_single_board_list(
                    single_board_list_train[i:i+batchsize])
                t1 = move_train[i:i+batchsize]
                #t2 = win_train[i:i+batchsize]
                time_mini_batch.end()

                time_train.start()
                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    # Logits for this minibatch
                    logits = model(x, training=True)

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(t1, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                sum_loss.append(loss_value)
                train_acc_metric.update_state(t1, logits)
                time_train.end()

                # print train loss and test accuracy
                if (step + 1) % eval_interval == 0:
                    time_val_mini_batch.start()
                    randomi = random.randint(0, max(0, test_num - batchsize))
                    x = make_input_features_from_single_board_list(
                        single_board_list_test[randomi:randomi+batchsize])
                    t1 = move_test[randomi:randomi+batchsize]
                    #t2 = win_test[i:i+batchsize]
                    y1 = model(x, training=False)
                    loss_value = loss_fn(t1, y1)
                    logger.info('epoch = {}, iteration = {}, loss = {}, accuracy = {}'.format(
                        e + 1, step + 1, np.average(sum_loss),
                        np.sum(np.sum(np.argmax(y1, axis=1) == t1)) / len(t1)))
                    sum_loss = []
                    time_val_mini_batch.end()

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            logger.info("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            time_val_epoch.start()
            # Run a validation loop at the end of each epoch.
            for step in range(0, test_num - batchsize, batchsize):

                x_batch_val = make_input_features_from_single_board_list(
                    single_board_list_test[step:step+batchsize])
                y_batch_val = move_test[step:step+batchsize]
                val_logits = model(x_batch_val, training=False)
                # Update val metrics
                val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            logger.info("Validation acc: %.4f" % (float(val_acc),))
            time_val_epoch.end()

        logging.info('save the model')
        model.save(save_model)

    logger.info(f'\n{TimeLog.debug()}')
