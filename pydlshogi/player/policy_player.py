import numpy as np
import scipy as sp
import shogi

from pydlshogi.common import *
from pydlshogi.features import *
from pydlshogi.player.base_player import *


# tensorflow by cpu

import os
if True:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras


def greedy(logits):
    return logits.index(max(logits))


def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return np.random.choice(len(logits), p=probabilities)


class PolicyPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.modelfile = '/Users/yanyano0207/Downloads/model_2017_policy/model/'
        self.model = None

    def usi(self):
        print('id name policy_player')
        print('option name modelfile type string default ' + self.modelfile)
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]

    def isready(self):
        if self.model is None:
            self.model = keras.models.load_model(self.modelfile)
        print('readyok')

    def position(self, moves):
        super().position(moves)
        self.position_calld = True

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return
        #print("turn", self.board.turn)
        features = make_input_features_from_board(self.board)
        logits = self.model.predict(features)[0]
        probabilities = sp.special.softmax(logits)

        # 全ての合法手について
        legal_moves = []
        legal_logits = []
        for move in self.board.legal_moves:
            # ラベルに変換
            label = make_output_label(move, self.board.turn)
            # 合法手とその指し手の確率(logits)を格納
            legal_moves.append(move)
            legal_logits.append(logits[label])
            # 確率を表示
            print('info string {:5} : {:.5f}'.format(
                move.usi(), probabilities[label]))

        # 確率が最大の手を選ぶ(グリーディー戦略)
        selected_index = greedy(legal_logits)
        # 確率に応じて手を選ぶ(ソフトマックス戦略)
        #selected_index = boltzmann(np.array(legal_logits, dtype=np.float32), 0.5)
        bestmove = legal_moves[selected_index]

        if self.position_calld:
            print('bestmove', bestmove.usi())
        self.position_calld = False
