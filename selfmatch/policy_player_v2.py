import shogi
import shogi.CSA
import game
import os
import sys
import itertools
import re
import glob

if False:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from pydlshogi_v2.features.features_v2 import FeaturesV2
import scipy as sp
import numpy as np
import csa_creater
import tensorflow as tf
from pydlshogi.time_log import TimeLog
from tensorflow import keras
from argparse import ArgumentParser

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


if __name__ == "__main__":
    print(sys.argv)
    time_kif = TimeLog("kif")

    class Standing:
        def __init__(self):
            self.win = 0
            self.lose = 0
            self.draw = 0
            self.win_rate = 0

        def addWin(self):
            self.win += 1
            self.win_rate = self.win / (self.win + self.lose)

        def addLose(self):
            self.lose += 1
            self.win_rate = self.win / (self.win + self.lose)

        def addDraw(self):
            self.draw += 1

    class StandingList():
        def __init__(self, name_list: 'list[str]' = None):
            self.standing_list: dict[str, Standing] = {}
            if name_list is not None:
                self.addPlayerNameList(name_list)

        def addPlayerName(self, name: str):
            self.standing_list[str(name)] = Standing()

        def addPlayerNameList(self, name_list: 'list[str]'):
            [self.addPlayerName(name) for name in name_list]

        def addResult(self, result: game.GameResult):
            if result.result == game.Result.BLACK_WIN:
                self.standing_list[result.black_player_name].addWin()
                self.standing_list[result.white_player_name].addLose()
            elif result.result == game.Result.WHITE_WIN:
                self.standing_list[result.black_player_name].addLose()
                self.standing_list[result.white_player_name].addWin()
            else:
                assert(result.result == game.Result.DRAW)
                self.standing_list[result.black_player_name].addDraw()
                self.standing_list[result.white_player_name].addDraw()

    parser = ArgumentParser()
    parser.add_argument('model_root', nargs='*')
    args = parser.parse_args()

    # Modelルートパスのリスト
    model_list = [os.path.abspath(path) for path in args.model_root]

    # チェックポイントのリスト
    ckpt_index_list = [glob.glob(os.path.join(
        model_root, 'weights*.ckpt.index')) for model_root in model_list]
    ckpt_index_list = list(itertools.chain.from_iterable(ckpt_index_list))

    # チェックポイントをlossの少ない順にソート
    ckpt_index_list.sort(key=lambda x: float(
        re.findall(r'(\d+\.\d+)\.ckpt\.index', x)[0]))

    player_list = []
#    for strategy in ['greedy', 'softmax']:
    for strategy in ['greedy']:
        for ckpt in ckpt_index_list:
            model_root = os.path.dirname(ckpt)
            player = PolicyPlayer(
                f'{strategy}_{os.path.basename(model_root)}_{os.path.basename(ckpt)}')
            player.setConfig(
                'ModelPath', os.path.join(model_root, 'model'))
            player.setConfig('WeightsPath', ckpt[:-6])
            player.setConfig('Strategy', strategy)
            try:
                player.prepare()
                player_list.append(player)
                print(player.name)
            except BaseException as e:
                print('error v2', model_root, os.path.basename(ckpt), e)
                raise e

    player_list += [game.Player()]
    player_match_list = [(player, player2)
                         for player in player_list for player2 in player_list if player != player2]

    standing_list = StandingList(player_list)
    for num, (player, player2) in enumerate(player_match_list):
        match = game.Game(player, player2)
        match.setDisplayKif(False)
        result = match.playMatch()

        time_kif.start()
        print(f'{num + 1}/{len(player_match_list)} {result.black_player_name} {result.white_player_name} {result.move_num} {result.result} {result.reason}')
        standing_list.addResult(result)

        ofile = f'{result.startTime.strftime("%Y%m%d-%H%M%S")}' \
            + f'_{result.black_player_name}' \
            + f'_{result.white_player_name}.csa'
        csa_creater.createKif(match.board, ofile, result)
        time_kif.end()

    for name, standing in sorted(standing_list.standing_list.items(), reverse=True,
                                 key=lambda x: x[1].win_rate):
        print(f'{name} {standing.win} - {standing.lose} - {standing.draw} {round(standing.win_rate * 100)}%')

    print(TimeLog.debug())
