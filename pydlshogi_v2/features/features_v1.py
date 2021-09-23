import pandas as pd
import shogi
import numpy as np
from pydlshogi_v2.features import position_list

# 移動の定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)

# 成り変換テーブル
MOVE_DIRECTION_PROMOTED = [
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
]

# 指し手を表すラベルの数
MOVE_DIRECTION_LABEL_NUM = len(MOVE_DIRECTION) + 7  # 7は持ち駒の種類

# rotate 180degree
SQUARES_R180 = [
    shogi.I1, shogi.I2, shogi.I3, shogi.I4, shogi.I5, shogi.I6, shogi.I7, shogi.I8, shogi.I9,
    shogi.H1, shogi.H2, shogi.H3, shogi.H4, shogi.H5, shogi.H6, shogi.H7, shogi.H8, shogi.H9,
    shogi.G1, shogi.G2, shogi.G3, shogi.G4, shogi.G5, shogi.G6, shogi.G7, shogi.G8, shogi.G9,
    shogi.F1, shogi.F2, shogi.F3, shogi.F4, shogi.F5, shogi.F6, shogi.F7, shogi.F8, shogi.F9,
    shogi.E1, shogi.E2, shogi.E3, shogi.E4, shogi.E5, shogi.E6, shogi.E7, shogi.E8, shogi.E9,
    shogi.D1, shogi.D2, shogi.D3, shogi.D4, shogi.D5, shogi.D6, shogi.D7, shogi.D8, shogi.D9,
    shogi.C1, shogi.C2, shogi.C3, shogi.C4, shogi.C5, shogi.C6, shogi.C7, shogi.C8, shogi.C9,
    shogi.B1, shogi.B2, shogi.B3, shogi.B4, shogi.B5, shogi.B6, shogi.B7, shogi.B8, shogi.B9,
    shogi.A1, shogi.A2, shogi.A3, shogi.A4, shogi.A5, shogi.A6, shogi.A7, shogi.A8, shogi.A9,
]


class FeaturesV1():
    def __init__(self):
        pass

    def positionListToFeature(self, data):
        single_board_list = np.asarray(data, dtype=np.int8)
        board_pieces = np.zeros(
            (single_board_list.shape[0], 81, (14+18+4+4+4+4+2+2)*2), dtype=np.int8)
        idx = 0
        hand_idx = 0
        for turn in [1, -1]:
            for piece in shogi.PIECE_TYPES:
                board_pieces[:, :, idx][data[:, :81] == (piece * turn)] = 1
                idx += 1
            for piece in range(shogi.PAWN, shogi.KING):
                for i in range(shogi.MAX_PIECES_IN_HAND[piece]):
                    board_pieces[:, :, idx][data[:, 81+hand_idx] > i] = 1
                    idx += 1
                hand_idx += 1

        return board_pieces.reshape(-1, 9, 9, 104)

    def boardToFeature(self, board: shogi.Board):
        return self.positionListToFeature(np.expand_dims(position_list.boardToSingleBoard(board), axis=0))

    def moveArrayListToLabel(self, data):
        return np.array([self.moveToLabel(d[0], d[1], d[2], d[3]) for d in data], dtype=np.int16)

    def moveToLabel(self, move_from, move_to, promotion, koma):
        # move direction
        if move_from >= 0:
            to_y, to_x = divmod(move_to, 9)
            from_y, from_x = divmod(move_from, 9)
            dir_x = to_x - from_x
            dir_y = to_y - from_y
            if dir_y < 0 and dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_LEFT
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_RIGHT
            elif dir_y < 0 and dir_x < 0:
                move_direction = UP_LEFT
            elif dir_y < 0 and dir_x > 0:
                move_direction = UP_RIGHT
            elif dir_y == 0 and dir_x < 0:
                move_direction = LEFT
            elif dir_y == 0 and dir_x > 0:
                move_direction = RIGHT
            elif dir_y > 0 and dir_x == 0:
                move_direction = DOWN
            elif dir_y > 0 and dir_x < 0:
                move_direction = DOWN_LEFT
            elif dir_y > 0 and dir_x > 0:
                move_direction = DOWN_RIGHT

            # promote
            if promotion:
                move_direction = MOVE_DIRECTION_PROMOTED[move_direction]
        else:
            # 持ち駒
            move_direction = len(MOVE_DIRECTION) + koma - 1

        move_label = 9 * 9 * move_direction + move_to

        return move_label


if __name__ == '__main__':
    from tensorflow import keras
    import pandas as pd
    import os
    model = keras.models.load_model('model_2017_policy_resnet2/model')
    df = pd.read_csv(os.path.join(
        os.environ['HOME'], 'data/floodgate/position_list_2020_minrate_4000_minmove_50.csv'))
    print(df.describe())

    feat = FeaturesV1()
    x = feat.positionListToFeature(
        df.loc[:, [c for c in df.columns if c[:2] == 'P_']].values)
    y = feat.moveArrayListToLabel(
        df.loc[:, [c for c in df.columns if c[:3] == 'MV_']].values)
    print(x.shape)
    print(np.asarray(y).shape)

    test_batch_size = 4096 * 4
    for i in range(10):
        try:
            print(model.evaluate(x, np.asarray(
                y, dtype=np.int16), batch_size=test_batch_size))
            print('test_batch_size', test_batch_size)
            break
        except BaseException as e:
            print(e)
            test_batch_size //= 2
