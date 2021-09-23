from argparse import ArgumentParser
import sys
import shogi
import shogi.CSA
import glob
from joblib import Parallel, delayed
import pandas as pd
import logging
import numpy as np

from shogi.Consts import BLACK
import itertools

KOMA_NAME = ['FU', 'KY', 'KE', 'GI', 'KI', 'KA', 'HI', 'OU'
             'TO', 'NY', 'NK', 'NG',       'UM', 'RY']


def boardToSingleBoard(board: shogi.Board):
    single_board = [board.piece_at(pos)
                    for pos in shogi.SQUARES]
    single_board = [0 if p is None else p.piece_type if p.color ==
                    shogi.BLACK else p.piece_type * -1 for p in single_board]
    piece_in_hands = [
        c[piece] if piece in c else 0 for c in board.pieces_in_hand for piece in shogi.PIECE_TYPES_WITHOUT_KING]
    return single_board + piece_in_hands


def moveToArray(move: shogi.Move, board: shogi.Board):
    from_sq = -1 if move.drop_piece_type else move.from_square
    to_sq = move.to_square
    promotion = 1 if move.promotion else 0
    koma = board.piece_type_at(
        from_sq) if move.drop_piece_type is None else move.drop_piece_type
    captured = 0 if board.piece_type_at(
        to_sq) is None else board.piece_type_at(to_sq)
    return [from_sq, to_sq, promotion, koma, captured]


def readCsaToPostion(kifu, fileinfo):
    try:
        kifu = shogi.CSA.Parser.parse_file(kifu)[0]
        board = shogi.Board()

        position_list = []
        for move_num, move in enumerate(kifu['moves']):
            single_board = boardToSingleBoard(board)
            move_array = moveToArray(shogi.Move.from_usi(move), board)
            board.push_usi(move)
            position_list.append(
                fileinfo + [move_num + 1] + single_board + move_array)

        return position_list
    except BaseException as e:
        logging.warning(e)
        raise e
        return []


def main(args):
    parser = ArgumentParser()
    parser.add_argument('kifulist')
    parser.add_argument('ofile')
    parser.add_argument('--min_rate', type=int, default=-1)
    parser.add_argument('--min_move_num', type=int, default=1)
    parser.add_argument('--max_num', type=int)
    args = parser.parse_args(args)

    # 棋譜リストを読み込み
    df = pd.read_csv(args.kifulist, index_col=0)

    # min_rateでフィルタリング
    df['both_min_rate'] = df.loc[:, ['black_rate', 'white_rate']].min(axis=1)
    df = df[df.both_min_rate >= args.min_rate]

    # move_numでフィルタリング
    df = df[df.move_num >= args.min_move_num]
    logging.info(df.describe())

    # レートが高い棋譜でフィルタリング
    if args.max_num:
        df = df.nlargest(args.max_num, 'both_min_rate')
    # boardオブジェクトに変換
    # Parallel(n_jobs=-1)(delayed(lambda i,x:(i,shogi.)))
    df.loc[:, 'win'].replace({'b': 1, '-': 0, 'w': -1}, inplace=True)

    position_list = [readCsaToPostion(
        df.at[index, 'filename'],
        [index, int(df.at[index, 'black_rate']), int(df.at[index, 'white_rate']), df.at[index, 'move_num']]) for index in df.index]
    position_list = itertools.chain.from_iterable(position_list)

    columns = ['FI_kif_index', 'FI_black_rate', 'FI_white_rate', 'FI_move_num']
    columns += ['MOVE_NUM']
    columns += [f'P_POS{r}{c}' for r in range(1, 10) for c in range(9, 0, -1)]
    columns += [f'P_HB{koma}' for koma in KOMA_NAME]
    columns += [f'P_HW{koma}' for koma in KOMA_NAME]
    columns += ['MV_FROM', 'MV_TO', 'MV_PM', 'MV_KOMA', 'MV_CAPT']
    df = pd.DataFrame(position_list, columns=columns)
    df.to_csv(args.ofile, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
