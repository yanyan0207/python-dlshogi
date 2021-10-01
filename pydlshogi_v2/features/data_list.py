from argparse import ArgumentParser
import sys
import shogi
import shogi.CSA
from joblib import Parallel, delayed
import pandas as pd
import logging
import numpy as np


def splitBoardRow(row: str):
    """
    盤上の1行を分割
    """
    ret = []
    while row:
        num = 2 if row[0] == '+' else 1
        ret.append(row[:num])
        row = row[num:]

    return ret


def reverseBoardRow(row: str):
    return ''.join(reversed(splitBoardRow(row)))


def reverseBoard(board: str):
    '''
    盤をひっくり返す
    '''
    return '/'.join(reversed([reverseBoardRow(row) for row in board.split('/')]))


def splitHands(hand_pieces: str):
    '''
    持ち駒を種類毎に分割
    '''

    piece_list = []
    while hand_pieces:
        for i, c in enumerate(hand_pieces):
            if c.isdecimal():
                continue
            piece_list.append(hand_pieces[:i+1])
            hand_pieces = hand_pieces[i+1:]
            break

    return piece_list


def sortHands(hand_pieces: str):
    return ''.join(sorted(splitHands(hand_pieces)))


def reversePos(pos: str):
    return str(10 - int(pos[0])) + chr(ord('a') + (ord('i') - ord(pos[1])))


def reverseMove(move: str):
    return ((move[:2] if move[1] == '*' else reversePos(move[:2]))
            + reversePos(move[2:4])
            + ('' if len(move) == 4 else move[4]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('sfenlist')
    parser.add_argument('ofile')
    parser.add_argument('--min_rate', type=int, default=-1)
    parser.add_argument('--min_move_num', type=int, default=1)
    parser.add_argument('--max_num', type=int)
    args = parser.parse_args()

    # 棋譜リストを読み込み
    df = pd.read_csv(args.sfenlist, index_col=0)

    # min_rateでフィルタリング
    df['both_min_rate'] = df.loc[:, [
        'FI_BLACK_RATE', 'FI_WHITE_RATE']].min(axis=1)
    df = df[df.both_min_rate >= args.min_rate]

    # move_numでフィルタリング
    df = df[df.FI_END_MOVE_NUM >= args.min_move_num]
    logging.info(df.describe())

    # レートが高い棋譜でフィルタリング
    if args.max_num:
        df = df.nlargest(args.max_num, 'both_min_rate')

    # 後手番の局面をひっくり返す
    df: pd.DataFrame
    df.loc[df.SFEN_TURN == 'w',
           'SFEN_BOARD'] = df.loc[df.SFEN_TURN == 'w', 'SFEN_BOARD'].apply(reverseBoard)
    df.loc[df.SFEN_TURN == 'w',
           'SFEN_BOARD'] = df.loc[df.SFEN_TURN == 'w', 'SFEN_BOARD'].str.swapcase()

    # 後手番の持ち駒をひっくり返す
    df.loc[df.SFEN_TURN == 'w',
           'SFEN_HANDS'] = df.loc[df.SFEN_TURN == 'w', 'SFEN_HANDS'].str.swapcase()

    # 後手番のMOVEをひっくり返す
    df.loc[df.SFEN_TURN == 'w', 'SFEN_MOVE'] = df.loc[df.SFEN_TURN ==
                                                      'w', 'SFEN_MOVE'].apply(reverseMove)

    # 持ち駒が一意になるようソート
    df['SFEN_HANDS'] = df['SFEN_HANDS'].apply(sortHands)
    df.to_csv("temp_sfen_turn.csv")

    # 勝率を算出
    df['win'] = (df.FI_END_RESULT == df.SFEN_TURN)
    df_win = df.groupby(['SFEN_BOARD', 'SFEN_HANDS']).mean()['win']
    # df_win.to_csv(args.ofile)

    df = df.groupby(['SFEN_BOARD', 'SFEN_HANDS']).agg(
        {'SFEN_MOVE': lambda x: x.value_counts().to_json(), 'win': np.mean})

    df.to_csv(args.ofile)
