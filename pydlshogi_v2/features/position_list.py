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

HAND_KOMA_NAME = ['FU', 'KY', 'KE', 'GI', 'KI', 'KA', 'HI']


def boardToSingleBoard(board: shogi.Board):
    single_board = [board.piece_at(pos) for pos in shogi.SQUARES]
    pieces_in_hand = board.pieces_in_hand
    if board.turn == shogi.WHITE:
        single_board = reversed(single_board)
        pieces_in_hand = reversed(pieces_in_hand)

    single_board = [0 if p is None
                    else p.piece_type * (1 if p.color == board.turn else -1)
                    for p in single_board]
    pieces_in_hand = [
        c[piece] if piece in c else 0 for c in pieces_in_hand
        for piece in range(shogi.PAWN, shogi.KING)]

    return single_board + pieces_in_hand


def moveToArray(move: shogi.Move, board: shogi.Board):
    from_sq = -1 if move.drop_piece_type else move.from_square
    to_sq = move.to_square
    promotion = 1 if move.promotion else 0
    koma = board.piece_type_at(
        from_sq) if move.drop_piece_type is None else move.drop_piece_type
    captured = 0 if board.piece_type_at(
        to_sq) is None else board.piece_type_at(to_sq)

    if board.turn == shogi.WHITE:
        from_sq = -1 if from_sq == -1 else 80 - from_sq
        to_sq = 80 - to_sq
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
                fileinfo +
                [move_num + 1] +
                single_board +
                move_array)

        return position_list
    except BaseException as e:
        logging.warning(e)
        raise e
        return []


def readPositionListCsv(path, min_rate=None, max_num=None, min_move_num=None):
    columns = pd.read_csv(path, index_col=0, nrows=0).columns.tolist()
    df = pd.read_csv(path, dtype={c:
                                  str if (c == 'FI_END_RESULT' or c == 'FI_END_REASON') else
                                  np.int32 if (c.startswith('FI'))
                                  else np.int8 for c in columns})

    # 手数でフィルタリング
    if min_move_num:
        df = df[df.FI_END_MOVE_NUM >= min_move_num]

    # レートでフィルタリング
    df['both_min_rate'] = df.loc[:, [
        'FI_BLACK_RATE', 'FI_WHITE_RATE']].min(axis=1)

    if min_rate:
        df = df[df.both_min_rate >= min_rate]

    # レートの高いデータを抽出
    if max_num:
        df = df.nlargest(max_num, 'both_min_rate')

    return df


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

    print(df.shape)
    # boardオブジェクトに変換
    position_list = Parallel(n_jobs=-1)(delayed(readCsaToPostion)(df.at[index, 'filename'],
                                                                  [index, int(df.at[index, 'black_rate']), int(df.at[index, 'white_rate']),
                                                                  df.at[index, 'move_num'], df.at[index, 'win'], df.at[index, 'end_reason']])
                                        for index in df.index)
#    position_list = [readCsaToPostion(
#        df.at[index, 'filename'],
#        [index, int(df.at[index, 'black_rate']), int(df.at[index, 'white_rate']), df.at[index, 'move_num']]) for index in df.index]
    position_list = itertools.chain.from_iterable(position_list)

    columns = ['FI_KIF_INDEX', 'FI_BLACK_RATE', 'FI_WHITE_RATE',
               'FI_END_MOVE_NUM', 'FI_END_RESULT', 'FI_END_REASON']
    columns += ['CURRENT_MOVE_NUM']
    columns += [f'P_POS{r}{c}' for r in range(1, 10) for c in range(9, 0, -1)]
    columns += [f'P_HB{koma}' for koma in HAND_KOMA_NAME]
    columns += [f'P_HW{koma}' for koma in HAND_KOMA_NAME]
    columns += ['MV_FROM', 'MV_TO', 'MV_PM', 'MV_KOMA', 'MV_CAPT']
    df = pd.DataFrame(position_list, columns=columns)
    df.to_csv(args.ofile, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
