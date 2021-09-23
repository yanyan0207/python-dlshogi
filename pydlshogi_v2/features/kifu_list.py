from argparse import ArgumentParser
import sys
import glob
import os
import shogi
import shogi.CSA
import logging
from joblib import Parallel, delayed
import pandas as pd


def read_csa(file):
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


def main(args):
    print(args)
    parser = ArgumentParser()
    parser.add_argument('kifu_root')
    parser.add_argument('ofile')
    args = parser.parse_args(args)

    # csaファイルの一覧
    csa_list = glob.glob(os.path.join(args.kifu_root, '*.csa'))

    # 読み込み
    board_list = Parallel(n_jobs=-1)(delayed(read_csa)(file)
                                     for file in csa_list)
    board_list = [b for b in board_list if b is not None]

    # 棋譜リストの作成
    df = pd.DataFrame(board_list)
    df.to_csv(args.ofile, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
