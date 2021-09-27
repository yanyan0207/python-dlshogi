from argparse import ArgumentParser
import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from pydlshogi_v2.features.features_v2 import FeaturesV2
from pydlshogi_v2.features.position_list import readPositionListCsv
from pydlshogi_v2.models import resnet_v2, modelUtil

import sys
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow_addons.optimizers import RectifiedAdam


def main(args):
    print(args)

    parser = ArgumentParser()
    parser.add_argument('train_position_list_csv')
    parser.add_argument('test_position_list_csv')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--train_min_rate', type=int)
    parser.add_argument('--train_max_num', type=int)
    parser.add_argument('--test_min_rate', type=int)
    parser.add_argument('--test_max_num', type=int)
    parser.add_argument('--min_move_num', type=int)
    parser.add_argument('--model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--moment', type=float, default=0.0)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--total_steps', type=int, default=0)

    args = parser.parse_args(args)
    df_train = readPositionListCsv(
        args.train_position_list_csv, min_move_num=args.min_move_num, min_rate=args.train_min_rate, max_num=args.train_max_num)
    df_test = readPositionListCsv(
        args.test_position_list_csv, min_move_num=args.min_move_num, min_rate=args.test_min_rate, max_num=args.test_max_num)

    print(df_train.shape)
    print(df_test.shape)
    pos_columns = [c for c in df_train.columns if c[:2] == 'P_']
    move_columns = [c for c in df_train.columns if c[:3] == 'MV_']

    features = FeaturesV2()

    train_ds = tf.data.Dataset.from_tensor_slices(
        (df_train.loc[:, pos_columns], features.moveArrayListToLabel(df_train.loc[:, move_columns].values)))
    test_ds = tf.data.Dataset.from_tensor_slices(
        (df_test.loc[:, pos_columns], features.moveArrayListToLabel(df_test.loc[:, move_columns].values)))

    train_ds = (train_ds
                .shuffle(len(train_ds))
                .batch(args.batch_size)
                .map(lambda x, y: ((tf.numpy_function(func=features.positionListToFeature, inp=[x], Tout=tf.int8),
                                   tf.numpy_function(func=features.handsToFeature, inp=[
                                                     x], Tout=tf.int8)),
                                   y))
                .prefetch(buffer_size=AUTOTUNE)
                )
    test_ds = (test_ds
               .batch(4096)
               .map(lambda x, y: ((tf.numpy_function(func=features.positionListToFeature, inp=[x], Tout=tf.int8),
                                   tf.numpy_function(func=features.handsToFeature, inp=[
                                                     x], Tout=tf.int8)),
                                  y))
               .prefetch(buffer_size=AUTOTUNE)
               )

    # チェックポイントコールバックを作る
    callbacks = []
    if args.model:
        callbacks.append(ModelCheckpoint(os.path.join(args.model, 'weights-{epoch:04d}-{val_loss:.2f}.ckpt'),
                                         save_weights_only=True,
                                         verbose=1))

    model = resnet_v2.createModel()

    if args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.learning_rate,
                        momentum=args.moment, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'radam':
        optimizer = RectifiedAdam(
            learning_rate=args.learning_rate, total_steps=args.total_steps)
    else:
        raise('unknown optimzer:' + args.optimizer)

    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])
    model.summary()

    history = model.fit(train_ds, epochs=args.epoch, batch_size=args.batch_size,
                        validation_data=test_ds, callbacks=callbacks, verbose=1)
    if args.model:
        model.save(os.path.join(args.model, 'model'))
        modelUtil.save_history(
            history, os.path.join(args.model, 'history.pickle'))
        modelUtil.save_history(
            history, os.path.join(args.model, 'history.csv'))


if __name__ == '__main__':
    main(sys.argv[1:])
