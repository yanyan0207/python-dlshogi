
import shogi.CSA
import game
import os
import sys
import itertools
import re
import glob
import csa_creater
import pandas as pd

from pydlshogi.time_log import TimeLog
from argparse import ArgumentParser

from selfmatch.policy_player_v2 import PolicyPlayer


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

    def addResultList(self, result_list):
        [self.addResult(result) for result in result_list]


def loadPlayer(model_root, ckpt_index, strategy):
    try:
        player = PolicyPlayer(
            f'{strategy}_{os.path.basename(model_root)}_{os.path.basename(ckpt_index)}')
        player.setConfig(
            'ModelPath', os.path.join(model_root, 'model'))
        player.setConfig('WeightsPath', ckpt_index[:-6])
        player.setConfig('Strategy', strategy)
        player.prepare()
    except BaseException as e:
        print('error v2', model_root, os.path.basename(ckpt_index), e)
        raise e
    return player


def loadPlayerList(model_root):
    model_root = os.path.abspath(model_root)
    ckpt_index_list = glob.glob(os.path.join(
        model_root, 'weights*.ckpt.index'))
    return [loadPlayer(model_root, ckpt_index, strategy)
            for strategy in ['greedy'] for ckpt_index in ckpt_index_list]


def gameMatch(num, allnum, player, player2, output_csa):
    match = game.Game(player, player2)
    match.setDisplayKif(False)
    result = match.playMatch()

    print(f'{num}/{allnum} {result.black_player_name} {result.white_player_name} {result.move_num} {result.result} {result.reason}')
    if output_csa:
        time_kif.start()
        ofile = f'{result.startTime.strftime("%Y%m%d-%H%M%S")}' \
            + f'_{result.black_player_name}' \
            + f'_{result.white_player_name}.csa'
        csa_creater.createKif(match.board, ofile, result)
        time_kif.end()
    return result


def leagMatch(player_list) -> StandingList:
    # 対戦リストを作成
    player_match_list = [(player, player2)
                         for player in player_list for player2 in player_list if player != player2]

    standing_list = StandingList(player_list)
    standing_list.addResultList([gameMatch(num+1, len(player_match_list), player, player2, args.output_csa)
                                 for num, (player, player2) in enumerate(player_match_list)])
    return standing_list


def outputModelMatchCsv(standing_list: StandingList, outfile):
    pd.DataFrame([[name, standing.win, standing.lose, standing.draw, standing.win_rate]
                 for name, standing in sorted(standing_list.standing_list.items())], columns=['name', 'win', 'lose', 'draw', 'win_rate']
                 ).to_csv(outfile, index=False)


if __name__ == "__main__":
    print(sys.argv)
    time_kif = TimeLog("kif")

    parser = ArgumentParser()
    parser.add_argument('model_root', nargs='*')
    parser.add_argument('--output_csa', action='store_true')
    args = parser.parse_args()

    # モデル内で戦う
    for model_root in args.model_root:
        model_match_csv = os.path.join(model_root, 'model_match.csv')
        if os.path.exists(model_match_csv):
            continue
        standing_list = leagMatch(loadPlayerList(model_root))
        outputModelMatchCsv(standing_list, model_match_csv)

    # モデル内の上位3人を取得
    player_list = []
    for model_root in args.model_root:
        model_match_csv = os.path.join(model_root, 'model_match.csv')
        df = pd.read_csv(model_match_csv, index_col=0)
        for name in df.nlargest(3, 'win_rate').index:
            strategy = name[:name.find('_')]
            ckpt_index = name[name.rfind('_') + 1:]
            player_list += [loadPlayer(model_root,
                                       os.path.join(model_root, ckpt_index), strategy)]

    player_list += [game.Player()]
    standing_list = leagMatch(player_list)

    for name, standing in sorted(standing_list.standing_list.items(), reverse=True,
                                 key=lambda x: x[1].win_rate):
        print(f'{name} {standing.win} - {standing.lose} - {standing.draw} {round(standing.win_rate * 100)}%')

    print(TimeLog.debug())
