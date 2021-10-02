from argparse import ArgumentParser
import numpy as np
from numpy.core.defchararray import isupper

from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.util.compat import as_str

from pydlshogi_v2.features.features_v2 import FeaturesV2
from pydlshogi_v2.features.position_list import boardToSingleBoard, readPositionListCsv
from pydlshogi_v2.models import resnet_v2, modelUtil

import sys
import os
import json
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow_addons.optimizers import RectifiedAdam

from joblib import Parallel, delayed

# 移動の定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

piece_index_list = {'P': 1, 'L': 2, 'N': 3, 'S': 4, 'G': 5, 'B': 6, 'R': 7, 'K': 8,
                    '+P': 9, '+L': 10, '+N': 11, '+S': 12, '+B': 13, '+R': 14}


def sfenBoardAndHandToSingleBoard(sfen_board, sfen_hand):
    single_board = np.zeros(81+14, dtype=np.int8)

    nari = ''
    index = 0
    for c in tf.compat.as_str_any(sfen_board):
        if c == '/':
            pass
        elif c.isdecimal():
            index += int(c)
        elif c == '+':
            nari = '+'
        else:
            piece = (1 if c.isupper() else -1) * \
                piece_index_list[f'{nari}{c.upper()}']
            single_board[index] = piece
            index += 1
            nari = ''

    num = 1
    for c in tf.compat.as_str_any(sfen_hand):
        if c == '-':
            break
        elif c.isdecimal():
            num = int(c)
        else:
            p_index = piece_index_list[c.upper()]
            single_board[
                (0 if c.isupper() else 7) + 81 + p_index - 1] = num

            num = 1
    return single_board


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
    else:
        assert(False)


def moveToLabel(move):
    if move[1] == '*':
        row, col = position(move[2:4])
        return (row * 9 + col) * 27 + 20 + piece_index_list[move[0].upper()] - 1
    else:
        fromrow, fromcol = position(move[0:2])
        torow, tocol = position(move[2:4])
        return (torow * 9 + tocol) * 27 + moveToDirection(torow - fromrow, tocol - fromcol) + (10 if len(move) == 5 else 0)


def sfenMoveToDictJson(moves):
    # json形式に変換
    data: dict = json.loads(tf.compat.as_str_any(moves))

    # 総計を計算
    all_nums = np.sum([num for num in data.values()])

    # すべての手の確率を返す
    return json.dumps({moveToLabel(move): num / all_nums for move, num in data.items()})


def sfenMoveToFeature(moves):
    move_labels = np.zeros(81*27, dtype=np.float32)

    # json形式に変換
    data: dict = json.loads(tf.compat.as_str_any(moves))

    # すべての手の確率を返す
    for move, num in data.items():
        move_labels[int(move)] = num

    return move_labels


def sfenMoveListToFeatures(moves_list):
    return np.asarray([sfenMoveToFeature(moves) for moves in moves_list])


def createDataSet(df: pd.DataFrame, batch_size, features: FeaturesV2, shuffle=True, only_toryo=True):
    def moveFuncWrapper(x):
        label = tf.numpy_function(sfenMoveListToFeatures, [x], tf.float32)
        label.set_shape(tf.TensorShape([None, 81*27]))
        return label

    print('sfenBoardAndHandToSingleBoard')
    # ポジションリスト
    position_list = np.asarray(Parallel(n_jobs=-1)(delayed(sfenBoardAndHandToSingleBoard)(sfen_board, sfen_hand)
                               for sfen_board, sfen_hand in zip(df['SFEN_BOARD'], df['SFEN_HANDS'])), dtype=np.int8)

    print('sfenMoveToDict')
    # 指し手リスト
    moves_dict_list = Parallel(
        n_jobs=-1)(delayed(sfenMoveToDictJson)(moves) for moves in df.SFEN_MOVE)

    print('from_tensor_slices')
    ds = tf.data.Dataset.from_tensor_slices(
        (position_list, moves_dict_list, df.win))

    ds = (ds
          .shuffle(len(ds) if shuffle else 1)
          .batch(batch_size)
          .map(lambda s, m, w: ((tf.numpy_function(func=features.positionListToFeature, inp=[s], Tout=tf.int8),
                                 tf.numpy_function(func=features.handsToFeature, inp=[s], Tout=tf.int8),),
                                (moveFuncWrapper(m), w)),
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

    model.compile(optimizer=optimizer, loss=[CategoricalCrossentropy(
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
