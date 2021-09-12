import argparse
import logging
import pandas as pd
import shogi
import shogi.CSA
import os


def read_csa(file, root_dir=None):
    try:
        kifu = shogi.CSA.Parser.parse_file(file)[0]
        if len(kifu['moves']) == 0:
            logging.warning(f'{file} no move')
        elif kifu['sfen'] != 'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1':
            logging.warning(f'{file} is not hirate')
        else:
            black_rate = -1
            white_rate = -1

            with open(file) as f:
                for line in f:
                    line = line.rstrip()
                    if line[:len("'black_rate")] == "'black_rate":
                        black_rate = float(line[line.rfind(':') + 1:])
                    elif line[:len("'white_rate")] == "'white_rate":
                        white_rate = float(line[line.rfind(':') + 1:])
                        break
                if root_dir is not None:
                    file = os.path.relpath(file, root_dir)
            return {
                'filename': file,
                'black_player': kifu['names'][0],
                'white_player': kifu['names'][1],
                'black_rate': black_rate,
                'white_rate': white_rate,
                'move_num': len(kifu['moves']),
                'win': kifu['win']
            }
    except ValueError as e:
        logging.warning(f"{file} parse_file error:{e}")


def add_argument_for_train(parser):
    parser.add_argument('train_kifu_list', help='train_kifu_list')
    parser.add_argument('test_kifu_list', help='test_kifu_list')
    parser.add_argument('--min_rate', type=float, help='min rate')
    parser.add_argument('--train_min_rate', type=float, help='train min rate')
    parser.add_argument('--test_min_rate', type=float, help='test min rate')
    parser.add_argument('--min_move_num', type=int, help='min move num')
    parser.add_argument('--train_max_num', type=int, help='train max num')
    parser.add_argument('--test_max_num', type=int, help='test max num')
    return parser


def load_kifulist(args):
    # CSVファイルの読み込み
    df_train = pd.read_csv(args.train_kifu_list, index_col=0)
    df_test = pd.read_csv(args.test_kifu_list, index_col=0)

    # 両プレーヤーの低い方のレートを計算
    df_train['both_min_rate'] = df_train.loc[:,
                                             ['black_rate', 'white_rate']].min(axis=1)
    df_test['both_min_rate'] = df_test.loc[:,
                                           ['black_rate', 'white_rate']].min(axis=1)

    # レートでフィルタリング
    train_min_rate = args.train_min_rate if args.train_min_rate is not None else args.min_rate
    test_min_rate = args.test_min_rate if args.test_min_rate is not None else args.min_rate

    if train_min_rate is not None:
        df_train = df_train[df_train.both_min_rate >= train_min_rate]
    if test_min_rate is not None:
        df_test = df_test[df_test.both_min_rate >= test_min_rate]

    # 手数でフィルタリング
    if args.min_move_num:
        df_train = df_train[df_train.move_num >= args.min_move_num]
        df_test = df_test[df_test.move_num >= args.min_move_num]

    # 最大数が設定されている時はレートの高い方からフィルタリング
    if args.train_max_num:
        df_train = df_train.nlargest(args.train_max_num, 'both_min_rate')
    if args.test_max_num:
        df_test = df_test.nlargest(args.test_max_num, 'both_min_rate')

    return df_train, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_argument_for_train(parser)
    args = parser.parse_args()
    df_train, df_test = load_kifulist(args)
    df_train.to_csv('df_train.csv')
    df_test.to_csv('df_test.csv')
