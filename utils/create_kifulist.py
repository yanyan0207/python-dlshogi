import argparse
import pandas as pd
import numpy as np
import os
import sys
import glob
import logging
from pydlshogi import kifulist

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

    kifu_list = [kifulist.read_csa(file, root_dir=root_dir)
                 for file in filelist]
    kifu_list = [kifu_info for kifu_info in kifu_list if kifu_info is not None]
    df = pd.DataFrame.from_dict(kifu_list)

    logging.info(f'file num {df.shape[0]}')
    df.to_csv(ofilename, index=False)
