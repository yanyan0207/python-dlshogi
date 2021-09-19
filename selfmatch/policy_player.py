import shogi
import shogi.CSA
import game
import os
from pydlshogi.features import *
import scipy as sp
import numpy as np
import glob
import math
import csa_creater

if True:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras


class PolicyPlayer(game.Player):
    def __init__(self, name='PolicyPlayer'):
        self.name = name
        self.modelPath = None
        self.weightsPath = None
        self.model: keras.models.model = None

    def setConfig(self, key, value):
        if key == 'ModelPath':
            self.modelPath = value
        elif key == 'WeightsPath':
            self.weightsPath = value

    def prepare(self):
        self.model = keras.models.load_model(self.modelPath)
        if self.weightsPath:
            self.model.load_weights(self.weightsPath)

    def startMatch(self):
        pass

    def bestMove(self, board: shogi.Board, legal_moves: 'set[shogi.Move]'):
        # 手がない場合は負け
        if len(legal_moves) == 0:
            return None

        features = make_input_features_from_board(board)
        logits = self.model.predict(features)[0]
        probabilities = sp.special.softmax(logits)
        moves = [(move, make_output_label(move, board.turn))
                 for move in legal_moves]
        probs = [probabilities[move_label] for move, move_label in moves]
        return moves[np.argmax(probs)][0]


if __name__ == "__main__":
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

    ckpt_index_list = glob.glob(
        '/Users/yanyano0207/Downloads/model_2017_policy/weights*.ckpt.index')
    player_list = [PolicyPlayer(os.path.basename(ckpt))
                   for ckpt in ckpt_index_list]

    for player, ckpt in zip(player_list, ckpt_index_list):
        player.setConfig(
            'ModelPath', '/Users/yanyano0207/Downloads/model_2017_policy/model/')
        player.setConfig('WeightsPath', ckpt[:-6])
        player.prepare()

    player_list += [game.Player()]
    player_match_list = [(player, player2)
                         for player in player_list for player2 in player_list if player != player2]

    standing_list = StandingList(player_list)
    for player, player2 in player_match_list:
        match = game.Game(player, player2)
        match.setDisplayKif(False)
        result = match.playMatch()
        print(result.black_player_name, result.white_player_name,
              result.move_num, result.result, result.reason)
        standing_list.addResult(result)
        ofile = f'{result.startTime.strftime("%Y%m%d-%H%M%S")}' \
            + f'_{result.black_player_name}' \
            + f'_{result.white_player_name}.csa'
        csa_creater.createKif(match.board, ofile, result)

    for name, standing in sorted(standing_list.standing_list.items(), reverse=True,
                                 key=lambda x: x[1].win_rate):
        print(f'{name} {standing.win} - {standing.lose} - {standing.draw} {round(standing.win_rate * 100)}%')
