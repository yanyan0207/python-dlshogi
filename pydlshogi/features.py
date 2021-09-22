import numpy as np
import shogi
import copy
import sys
import pandas as pd

from shogi.Consts import BLACK
from pydlshogi.common import *
from pydlshogi.time_log import *

time_board_pieces = TimeLog("borad pieces")
time_pieces_in_hand = TimeLog("pieces in hand")


def add_argument_for_train(parser):
    parser.add_argument('train_position_list', help='train_position_list')
    parser.add_argument('test_position_list', help='test_position_list')
    parser.add_argument('--min_rate', type=float, help='min rate')
    parser.add_argument('--train_min_rate', type=float, help='train min rate')
    parser.add_argument('--test_min_rate', type=float, help='test min rate')
    parser.add_argument('--min_move_num', type=int, help='min move num')
    parser.add_argument('--train_max_num', type=int, help='train max num')
    parser.add_argument('--test_max_num', type=int, help='test max num')
    return parser


def load_positionlist(args):
    dtypes = {'A_black_rate': np.int16, 'A_white_rate': np.int16,
              'A_move_num': np.int16, 'F_current_movenum': np.int16}
    for i in range(9*9):
        dtypes[f'F_pos{i//9+1}{i%9+1}'] = np.int8
    for koma in ['fu', 'ky', 'ke', 'gi', 'ki', 'ka', 'hi']:
        dtypes[f'F_b{koma}'] = np.int8
        dtypes[f'F_w{koma}'] = np.int8
    dtypes['L_hand'] = np.int16
    dtypes['L_win'] = np.int8

    # レートでフィルタリング
    train_min_rate = args.train_min_rate if args.train_min_rate is not None else args.min_rate
    test_min_rate = args.test_min_rate if args.test_min_rate is not None else args.min_rate

    df_list = []
    for position_list, min_rate, max_num in [
        (args.train_position_list, train_min_rate, args.train_max_num),
            (args.test_position_list, test_min_rate, args.test_max_num)]:
        df = pd.read_csv(position_list, index_col=0, dtype=dtypes)

        # 両プレーヤーの低い方のレートを計算
        df['both_min_rate'] = df.loc[:, [
            'A_black_rate', 'A_white_rate']].min(axis=1)

        if min_rate is not None:
            df = df[df.both_min_rate >= min_rate]

        # 手数でフィルタリング
        if args.min_move_num:
            df = df[df.A_move_num >= args.min_move_num]

        # 最大数が設定されている時はレートの高い方からフィルタリング
        if max_num:
            df = df.nlargest(max_num, 'both_min_rate')

        df_list.append(df)
    return df_list


def posions_to_single_board(positions):
    return [posion_to_single_board(position) for position in positions]


def board_to_single_board(piece_bb, occupied, pieces_in_hand):
    single_board_and_piece_in_hand = np.zeros(9*9 + 7*2, dtype=np.int8)

    for color in shogi.COLORS:
        # board pieces
        time_board_pieces.start()
        for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]:
            bb = piece_bb[piece_type] & occupied[color]
            feature = np.zeros(9*9, dtype=np.int8)
            for pos in shogi.SQUARES:
                if bb & shogi.BB_SQUARES[pos] > 0:
                    single_board_and_piece_in_hand[pos] = piece_type * \
                        (1 if color == shogi.BLACK else -1)
    idx = 81
    for color in shogi.COLORS:
        # pieces in hand
        for piece_type in range(1, 8):
            for n in range(shogi.MAX_PIECES_IN_HAND[piece_type]):
                if piece_type in pieces_in_hand[color]:
                    single_board_and_piece_in_hand[idx] = pieces_in_hand[color][piece_type]
            idx += 1

    return single_board_and_piece_in_hand


def posion_to_single_board(position):
    piece_bb, occupied, pieces_in_hand, move, win = position
    single_board_and_piece_in_hand = np.zeros(9*9 + 7*2, dtype=np.int8)

    for color in shogi.COLORS:
        # board pieces
        time_board_pieces.start()
        for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]:
            bb = piece_bb[piece_type] & occupied[color]
            feature = np.zeros(9*9, dtype=np.int8)
            for pos in shogi.SQUARES:
                if bb & shogi.BB_SQUARES[pos] > 0:
                    single_board_and_piece_in_hand[pos] = piece_type * \
                        (1 if color == shogi.BLACK else -1)
    idx = 81
    for color in shogi.COLORS:
        # pieces in hand
        for piece_type in range(1, 8):
            for n in range(shogi.MAX_PIECES_IN_HAND[piece_type]):
                if piece_type in pieces_in_hand[color]:
                    single_board_and_piece_in_hand[idx] = pieces_in_hand[color][piece_type]
            idx += 1

    return single_board_and_piece_in_hand, move, win


def make_input_features_from_single_board_list(single_board_list):
    single_board_list = np.asarray(single_board_list, dtype=np.int8)
    sample_num = single_board_list.shape[0]
    features = np.zeros(
        (sample_num, 81, (14+7)*2), dtype=np.int8)

    idx = 0
    in_hand_idx = 81
    for color in shogi.COLORS:
        # board pieces
        for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]:
            features[:, :, idx][single_board_list[:, :81] ==
                                piece_type * (1 if color == shogi.BLACK else -1)] = 1
            idx += 1
        # pieces in hand
        for piece_type in range(1, 8):
            features[:, :, idx] = np.expand_dims(single_board_list[:, in_hand_idx], axis=-1) / \
                shogi.MAX_PIECES_IN_HAND[piece_type]
            idx += 1
            in_hand_idx += 1
    return features.reshape(sample_num, 9, 9, 42)


def make_input_features(piece_bb, occupied, pieces_in_hand):
    features = []
    for color in shogi.COLORS:
        # board pieces
        time_board_pieces.start()
        for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]:
            bb = piece_bb[piece_type] & occupied[color]
            feature = np.zeros(9*9, dtype=np.int8)
            for pos in shogi.SQUARES:
                if bb & shogi.BB_SQUARES[pos] > 0:
                    feature[pos] = 1
            features.append(feature.reshape((9, 9)))
        time_board_pieces.end()

        # pieces in hand
        time_pieces_in_hand.start()
        for piece_type in range(1, 8):
            for n in range(shogi.MAX_PIECES_IN_HAND[piece_type]):
                if piece_type in pieces_in_hand[color] and n < pieces_in_hand[color][piece_type]:
                    feature = np.ones(9*9, dtype=np.int8)
                else:
                    feature = np.zeros(9*9, dtype=np.int8)
                features.append(feature.reshape((9, 9)))
        time_pieces_in_hand.end()

    return features


def make_input_features_from_board(board):
    if board.turn == shogi.BLACK:
        piece_bb = board.piece_bb
        occupied = (board.occupied[shogi.BLACK], board.occupied[shogi.WHITE])
        pieces_in_hand = (
            board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE])
    else:
        piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
        occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(
            board.occupied[shogi.BLACK]))
        pieces_in_hand = (
            board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK])

    single_board = board_to_single_board(piece_bb, occupied, pieces_in_hand)
    return make_input_features_from_single_board_list([single_board])


def make_output_label(move, color):
    move_to = move.to_square
    move_from = move.from_square

    # 白の場合盤を回転
    if color == shogi.WHITE:
        move_to = SQUARES_R180[move_to]
        if move_from is not None:
            move_from = SQUARES_R180[move_from]

    # move direction
    if move_from is not None:
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
        if move.promotion:
            move_direction = MOVE_DIRECTION_PROMOTED[move_direction]
    else:
        # 持ち駒
        move_direction = len(MOVE_DIRECTION) + move.drop_piece_type - 1

    move_label = 9 * 9 * move_direction + move_to

    return move_label


def make_features(position):
    piece_bb, occupied, pieces_in_hand, move, win = position
    features = make_input_features(piece_bb, occupied, pieces_in_hand)

    return (features, move, win)
