import shogi
import shogi.CSA
import game
import os

if True:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from pydlshogi_v2.features.features_v2 import FeaturesV2
import scipy as sp
import numpy as np

import tensorflow as tf
from pydlshogi.time_log import TimeLog
from tensorflow import keras


time_feature = TimeLog('feature')
time_predict = TimeLog('predict')
time_softmax = TimeLog('softmax')
time_label = TimeLog('label')
time_logits = TimeLog('logits')
time_find_max = TimeLog('find_max')


class PolicyPlayer(game.Player):
    def __init__(self, name='PolicyPlayer'):
        self.name = name
        self.modelPath = None
        self.weightsPath = None
        self.model: keras.models.model = None
        self.strategy = 'greedy'
        self.features = None

    def setConfig(self, key, value):
        if key == 'ModelPath':
            self.modelPath = value
        elif key == 'WeightsPath':
            self.weightsPath = value
        elif key == 'Strategy':
            self.strategy = value

    def prepare(self):
        model = keras.models.load_model(self.modelPath)
        if self.weightsPath:
            model.load_weights(self.weightsPath)
        model.call = tf.function(
            model.call, experimental_relax_shapes=True)
        self.model = model
        self.features = FeaturesV2()
        board = shogi.Board()
        self.bestMove(board, {move for move in board.legal_moves})

    def startMatch(self):
        pass

    def bestMove(self, board: shogi.Board, legal_moves: 'set[shogi.Move]'):
        # 手がない場合は負け
        if len(legal_moves) == 0:
            return None

        # ボードから特長量を取得
        time_feature.start()
        features = self.features.boardToFeature(board)
        time_feature.end()

        # policy予測
        time_predict.start()
        pred_logits = np.array(self.model(features, training=False)[0][0])
        time_predict.end()

        # 合法手の一覧をラベルに変換
        time_label.start()
        legal_moves = list(legal_moves)
        logical_move_labels = [self.features.boardMoveToLabel(
            move, board) for move in legal_moves]
        time_label.end()

        # 合法種の一覧の予測値を取得
        time_logits.start()
        pred_logits_in_logical_moves = [pred_logits[label]
                                        for label in logical_move_labels]
        time_logits.end()

        if self.strategy == 'greedy':
            time_find_max.start()
            best_move = legal_moves[np.argmax(pred_logits_in_logical_moves)]
            time_find_max.end()
        elif self.strategy == 'softmax':
            time_softmax.start()
            probabilities = sp.special.softmax(pred_logits_in_logical_moves)
            time_softmax.end()
            time_find_max.start()
            best_move = legal_moves[np.random.choice(
                len(probabilities), p=probabilities)]
            time_find_max.end()
        else:
            raise RuntimeError(f'Unkown Strategy:{self.strategy}')

        time_find_max.end()
        return best_move
