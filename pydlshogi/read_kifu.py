import shogi
import shogi.CSA
import copy
import os

from pydlshogi.features import *

# read kifu


def read_kifu(kifu_list_file, root=None):
    with open(kifu_list_file, 'r') as f:
        return read_kifu_array([line.rstrip('\r\n') for line in f.readlines()])


def read_kifu_array(kifu_list, root=None):
    positions = []
    for filepath in kifu_list:
        positions.extend(read_kifu_single(filepath, root))
    return positions


def read_kifu_single(filepath, root=None):
    if root is not None:
        filepath = os.path.join(root, filepath)
    kifu = shogi.CSA.Parser.parse_file(filepath)[0]
    win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE
    board = shogi.Board()
    positions = []
    for move in kifu['moves']:
        if board.turn == shogi.BLACK:
            piece_bb = copy.deepcopy(board.piece_bb)
            occupied = copy.deepcopy(
                (board.occupied[shogi.BLACK], board.occupied[shogi.WHITE]))
            pieces_in_hand = copy.deepcopy(
                (board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE]))
        else:
            piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
            occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(
                board.occupied[shogi.BLACK]))
            pieces_in_hand = copy.deepcopy(
                (board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK]))

        # move label
        move_label = make_output_label(
            shogi.Move.from_usi(move), board.turn)

        # result
        win = 1 if win_color == board.turn else 0

        positions.append(
            (piece_bb, occupied, pieces_in_hand, move_label, win))
        board.push_usi(move)
    return positions
