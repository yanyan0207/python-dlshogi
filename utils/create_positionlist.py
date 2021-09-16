import argparse
from pydlshogi.features import posions_to_single_board
from pydlshogi.read_kifu import read_kifu, read_kifu_single
import pandas as pd
from joblib import Parallel, delayed
import os
from pathlib import Path
import tempfile
import glob


def create_feature(index, kifu,  black_rate, white_rate, root, odir, columns):
    positions = posions_to_single_board(read_kifu_single(kifu, root))
    # ファイルidx,手番レート、相手レート、手数、シングルボード＋持ち駒、打ち手、勝利者
    features = []
    move_num = len(positions)
    for i, (single_board, move, win) in enumerate(positions):
        if i % 2 == 0:
            feature = [index, black_rate, white_rate, move_num, i+1]
        else:
            feature = [index, white_rate, black_rate, move_num, i+1]
        feature.extend(list(single_board))
        feature.extend([move, win])
        features.append(feature)

    df = pd.DataFrame(columns=columns, data=features)
    ofilename = os.path.join(odir, f'{index:08}_{Path(kifu).stem}.csv')
    df.to_csv(ofilename, index=False)
    return ofilename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('kifulist')
    parser.add_argument('ofile')
    parser.add_argument('--kifu_root')
    parser.add_argument('--min_rate', type=float)
    parser.add_argument('--min_move_num', type=int)
    parser.add_argument('--max_num', type=int)

    args = parser.parse_args()
    df_kifulist = pd.read_csv(args.kifulist)
    df_kifulist['both_min_rate'] = df_kifulist.loc[:,
                                                   ['black_rate', 'white_rate']].min(axis=1)

    if args.min_rate:
        df_kifulist = df_kifulist[df_kifulist['both_min_rate']
                                  >= args.min_rate]

    if args.min_move_num:
        df_kifulist = df_kifulist[df_kifulist['move_num'] >= args.min_move_num]

    if args.max_num:
        df_kifulist = df_kifulist.nlargest(args.max_num, 'both_min_rate')

    print(df_kifulist.head())

    columns = ["file_index", "A_black_rate", "A_white_rate",
               "A_move_num", "F_current_movenum"]
    for i in range(81):
        columns.append(f"F_pos{9 - i % 9}{i // 9 + 1}")
    columns.extend(['F_bfu', 'F_bky', 'F_bke',
                   'F_bgi', 'F_bki', 'F_bka', 'F_bhi'])
    columns.extend(['F_wfu', 'F_wky', 'F_wke',
                   'F_wgi', 'F_wki', 'F_wka', 'F_whi'])
    columns.extend(['L_hand', 'L_win'])

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1局ごとに、1行1手の情報が書かれているCSVを作成
        ofilelist = Parallel(n_jobs=-1)(
            delayed(create_feature)(idx, filename, int(black_rate),
                                    int(white_rate), args.kifu_root, tmpdir, columns)
            for idx, filename, black_rate, white_rate in zip(df_kifulist.index, df_kifulist.filename, df_kifulist.black_rate, df_kifulist.white_rate))

        # 全てのファイルを連結
        filelist = sorted(glob.glob(f'{tmpdir}/*.csv'))
        if len(filelist) == 0:
            raise RuntimeError("no kifu")

        with open(args.ofile, mode='x') as ofile:
            # ヘッダ行を書き込む
            with open(filelist[0]) as f:
                ofile.write(f.readline())

            # 全ての行を書き込む
            for path in filelist:
                with open(path) as f:
                    ofile.writelines(f.readlines()[1:])
