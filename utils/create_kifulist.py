import argparse
import pandas as pd
import numpy as np
import os
import sys
import glob
import logging
import shogi
import shogi.CSA


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


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("kifu_dir")
    parser.add_argument("ofilename")
    parser.add_argument("--recursive", "-r", action="store_true")
    parser.add_argument("--root_dir")
    args = parser.parse_args()

    kifu_dir = args.kifu_dir
    recursive = args.recursive
    root_dir = args.root_dir
    ofilename = args.ofilename

    if recursive:
        filelist = glob.glob(f"{kifu_dir}/**/*.csa", recursive=True)
    else:
        filelist = glob.glob(f"{kifu_dir}/*.csa")

    if len(filelist) == 0:
        logging.error("no file")
        sys.exit(1)

    kifu_list = [read_csa(file, root_dir=root_dir) for file in filelist]
    kifu_list = [kifu_info for kifu_info in kifu_list if kifu_info is not None]
    df = pd.DataFrame.from_dict(kifu_list)

    logging.info(f'file num {df.shape[0]}')
    df.to_csv(ofilename)
