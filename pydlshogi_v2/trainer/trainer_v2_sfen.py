from argparse import ArgumentParser
import numpy as np
from numpy.core.defchararray import isupper

from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from pydlshogi_v2.features.features_v2 import FeaturesV2
from pydlshogi_v2.features.position_list import readPositionListCsv
from pydlshogi_v2.models import resnet_v2, modelUtil

import sys
import os
import json
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow_addons.optimizers import RectifiedAdam

# 移動の定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

piece_index_list = {'P': 0, 'L': 1, 'N': 2,
                    'S': 3, 'B': 4, 'R': 5, 'G': 6, 'K': 7}
mochikoma_max_list = [18, 4, 4, 4, 2, 2, 4]
mochikoma_start_list = [0, 18, 22, 26, 30, 32, 34, 38]


def sfenBoardToFeature(sfen_board_list):
    board_list = np.zeros((sfen_board_list.shape[0], 9, 9, 28), dtype=np.int8)

    for sfen_index, sfen_board in enumerate(sfen_board_list):
        sfen_board: str
        row = 0
        col = 0
        nari = False
        for c in tf.compat.as_str_any(sfen_board):
            if c == '/':
                row += 1
                col = 0
            elif c.isdecimal():
                col += int(c)
            elif c == '+':
                nari = True
            else:
                piece = (0 if c.isupper() else 14) + \
                    (8 if nari else 0) + piece_index_list[c.upper()]
                board_list[sfen_index, row, col, piece] += 1
                col += 1
                nari = False

    return board_list


def sfenHandsToFeature(sfen_hand_list):
    hand_list = np.zeros((sfen_hand_list.shape[0], 76), dtype=np.int8)

    BREAK_FLAG = False
    for sfen_index, sfen_hand in enumerate(sfen_hand_list):
        sfen_hand: str

        num = 1
        for c in tf.compat.as_str_any(sfen_hand):
            if c == '-':
                break
            elif c.isdecimal():
                num = int(c)
            else:
                p_index = piece_index_list[c.upper()]
                for i in range(num):
                    hand_list[sfen_index,
                              (0 if c.isupper() else mochikoma_start_list[-1]) +
                              mochikoma_start_list[p_index] + i] += 1

                num = 1
        if BREAK_FLAG:
            BREAK_FLAG = False
            break
    return hand_list


def position(pos):
    return ord(pos[1]) - ord('a'), 9 - int(pos[0])


def moveToDirection(rowmove, colmove):
    if rowmove == colmove:
        return DOWN_RIGHT if rowmove > 0 else UP_LEFT
    elif rowmove * -1 == colmove:
        return DOWN_LEFT if rowmove > 0 else UP_RIGHT
    elif rowmove == 0:
        return RIGHT if colmove > 0 else LEFT
    elif colmove == 0:
        return DOWN if rowmove > 0 else UP
    elif rowmove == -2:
        return UP2_RIGHT if colmove > 0 else UP2_LEFT


def moveToLabel(move):
    if move[1] == '*':
        row, col = position(move[2:4])
        return (row * 9 + col) * 27 + 20 + piece_index_list[move[0].upper()]
    else:
        fromrow, fromcol = position(move[0:2])
        torow, tocol = position(move[2:4])
        return (torow * 9 + tocol) * 27 + moveToDirection(torow - fromrow, tocol - fromcol) + (10 if len(move) == 5 else 0)


def sfenMoveToFeature(moves_list):
    move_label_list = np.zeros(moves_list.shape[0], dtype=np.int16)
    for index, moves in enumerate(moves_list):
        data = json.loads(moves)
        move_list, num_list = list(zip(*[(k, v) for k, v in data.items()]))

        move = move_list[np.argmax(num_list)]
        move_label_list[index] = moveToLabel(move)
    return move_label_list


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def createDataSet(df, batch_size, features: FeaturesV2, shuffle=True, only_toryo=True):
    win = df.win
    moves = sfenMoveToFeature(df.SFEN_MOVE)
    ds = tf.data.Dataset.from_tensor_slices(
        (df.SFEN_BOARD, df.SFEN_HANDS, (moves, win)))

    ds = (ds
          .shuffle(len(ds) if shuffle else 1)
          .batch(batch_size)
          .map(lambda b, h, y: ((tf.numpy_function(func=sfenBoardToFeature, inp=[b], Tout=tf.int8),
                                 tf.numpy_function(func=sfenHandsToFeature, inp=[h], Tout=tf.int8)),
                                y)
               )
          .prefetch(buffer_size=AUTOTUNE)
          )
    return ds


def main(args):
    print(args)

    parser = ArgumentParser()
    parser.add_argument('train_position_list_csv')
    parser.add_argument('test_position_list_csv')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--moment', type=float, default=0.0)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--total_steps', type=int, default=0)

    args = parser.parse_args(args)
    columns = pd.read_csv(args.train_position_list_csv,
                          nrows=0).columns.tolist()
    df_train = pd.read_csv(args.train_position_list_csv, dtype={
                           c: float if c == 'win' else str for c in columns})
    df_test = pd.read_csv(args.test_position_list_csv, dtype={
        c: float if c == 'win' else str for c in columns})
    df_train['win'] = df_train['win'].astype(np.int8)
    df_test['win'] = df_test['win'].astype(np.int8)

    print(df_train.shape)
    print(df_test.shape)

    features = FeaturesV2()

    train_ds = createDataSet(df_train, args.batch_size, features)
    test_ds = createDataSet(df_test, 4096, features)

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

    model.compile(optimizer=optimizer, loss=[SparseCategoricalCrossentropy(
        from_logits=True), BinaryCrossentropy(
        from_logits=True)], metrics='accuracy')
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
