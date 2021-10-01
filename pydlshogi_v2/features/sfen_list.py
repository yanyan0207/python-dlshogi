from argparse import ArgumentParser
import sys
import shogi
import shogi.CSA
from joblib import Parallel, delayed
import pandas as pd
import logging
import numpy as np


def readCsaToSfensInMatch(kifu):
    try:
        kifu = shogi.CSA.Parser.parse_file(kifu)[0]
        board = shogi.Board()

        # 全ての手に対して
        sfens = []
        for move in kifu['moves']:
            # Sfenをリストに追加
            sfens.append((board.sfen(), move))

            # 一手進める
            board.push_usi(move)

        return sfens
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

    # CSAファイルから全ての手番のSFENを取得
    sfens_list = Parallel(n_jobs=-1)(delayed(readCsaToSfensInMatch)
                                     (df.at[index, 'filename']) for index in df.index)

    # ファイル情報を作成
    file_info_list = [[index,
                       int(df.at[index, 'black_rate']),
                       int(df.at[index, 'white_rate']),
                       df.at[index, 'move_num'],
                       df.at[index, 'win'],
                       df.at[index, 'end_reason']] for index in df.index]

    # 全ての対局の全ての手に対してデータを作成する
    data_list = [file_info + sfen.split(' ') + [move] for file_info, sfens in zip(
        file_info_list, sfens_list) for sfen, move in sfens]

    # ヘッダの作成
    columns = ['FI_KIF_INDEX', 'FI_BLACK_RATE', 'FI_WHITE_RATE',
               'FI_END_MOVE_NUM', 'FI_END_RESULT', 'FI_END_REASON']
    columns += ['SFEN_BOARD', 'SFEN_TURN', 'SFEN_HANDS', 'SFEN_MOVE_NUM']
    columns += ['SFEN_MOVE']
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(args.ofile, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
