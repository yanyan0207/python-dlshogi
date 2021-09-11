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

import logging


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


def createModel():
    ch = 192
    inputs = keras.Input(shape=(9, 9, 104), name="digits")
    x = Conv2D(ch, kernel_size=(3, 3),
               activation='relu', padding='same')(inputs)
    for i in range(11):
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

# mini batch


def mini_batch(positions, i, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)

    return (np.array(mini_batch_data, dtype=np.float32).transpose(0, 2, 3, 1),
            np.array(mini_batch_move, dtype=np.int32),
            np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))


def mini_batch_for_test(positions, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = make_features(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)

    return (np.array(mini_batch_data, dtype=np.float32).transpose(0, 2, 3, 1),
            np.array(mini_batch_move, dtype=np.int32),
            np.array(mini_batch_win, dtype=np.int32).reshape((-1, 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('kifulist_train', type=str, help='train kifu list')
    parser.add_argument('kifulist_test', type=str, help='test kifu list')
    parser.add_argument('--kifulist_root')
    parser.add_argument('--blocks', type=int, default=5,
                        help='Number of resnet blocks')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of positions in each mini-batch')
    parser.add_argument('--test_batchsize', type=int, default=512,
                        help='Number of positions in each test mini-batch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=1, help='Number of epoch times')
    parser.add_argument(
        '--model', type=str, default='model/model_policy', help='model file name')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--log', default=None, help='log file path')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--eval_interval', '-i', type=int,
                        default=1000, help='eval interval')
    args = parser.parse_args()
    epochs = 2
    batchsize = args.batchsize
    lr = args.lr
    eval_interval = args.eval_interval
    initmodel = args.initmodel
    save_model = args.model

    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

    logger = logging.getLogger()
    if args.log is None:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    logger.info('read kifu start')
    # 保存済みのpickleファイルがある場合、pickleファイルを読み込む
    # train date
    time_kifulist_train = TimeLog('kifulist_train')
    time_kifulist_train.start()
    train_pickle_filename = f"{Path(args.kifulist_train).stem}.pickle"
    if os.path.exists(train_pickle_filename):
        with open(train_pickle_filename, 'rb') as f:
            positions_train = pickle.load(f)
        logger.info('load train pickle')
    else:
        positions_train = read_kifu(
            args.kifulist_train, root=args.kifulist_root)
    time_kifulist_train.end()

    # test data
    time_kifulist_test = TimeLog('kifulist_test')
    time_kifulist_test.start()
    test_pickle_filename = f"{Path(args.kifulist_test).stem}.pickle"
    if os.path.exists(test_pickle_filename):
        with open(test_pickle_filename, 'rb') as f:
            positions_test = pickle.load(f)
        logger.info('load test pickle')
    else:
        positions_test = read_kifu(args.kifulist_test, root=args.kifulist_root)
    time_kifulist_test.end()

    # 保存済みのpickleがない場合、pickleファイルを保存する
    time_pickle = TimeLog('pickle')
    time_pickle.start()
    if not os.path.exists(train_pickle_filename):
        with open(train_pickle_filename, 'wb') as f:
            pickle.dump(positions_train, f, pickle.HIGHEST_PROTOCOL)
        logger.info('save train pickle')
    if not os.path.exists(test_pickle_filename):
        with open(test_pickle_filename, 'wb') as f:
            pickle.dump(positions_test, f, pickle.HIGHEST_PROTOCOL)
        logger.info('save test pickle')
    logger.info('read kifu end')
    time_pickle.end()

    logger.info('train position num = {}'.format(len(positions_train)))
    logger.info('test position num = {}'.format(len(positions_test)))

    # train
    time_mini_batch = TimeLog("mini_batch")
    time_train = TimeLog("train")
    time_val_mini_batch = TimeLog("val_mini_batch")
    time_val_epoch = TimeLog("val_epoch")

    logger.info('start training')

    model = createModel()
    model_summary = []
    model.summary(
        print_fn=lambda x: model_summary.append(x))
    logger.info("\n".join(model_summary))

    # Init/Resume
    if initmodel:
        logging.info('Load model from {}'.format(args.initmodel))
        keras.models.load_model("initmodel")
    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=lr)

    # Instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Prepare the metrics.
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # train
    logger.info('start training')
    for e in range(epochs):
        logger.info("\nStart of epoch %d" % (e+1,))

        positions_train_shuffled = random.sample(
            positions_train, len(positions_train))

        sum_loss = []
        for step, i in enumerate(range(0, len(positions_train_shuffled) - batchsize, batchsize)):
            time_mini_batch.start()
            x, t1, t2 = mini_batch(
                positions_train_shuffled, i, batchsize)
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
                x, t1, t2 = mini_batch_for_test(
                    positions_test, args.test_batchsize)
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
        for step in range(0, len(positions_test) - batchsize, batchsize):
            x_batch_val, y_batch_val, _ = mini_batch(
                positions_test, step, batchsize)
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        logger.info("Validation acc: %.4f" % (float(val_acc),))
        time_val_epoch.end()

    logging.info('save the model')
    model.save(save_model)

    logger.info(TimeLog.debug())
