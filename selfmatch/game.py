from os import name
import shogi
import copy
import random
from enum import Enum
from shogi.Consts import BLACK, WHITE
from datetime import datetime as dt


class Player():
    def __init__(self, name='Player'):
        self.name = name

    def __str__(self):
        return self.name

    def setConfig(self, key, value):
        pass

    def prepare(self):
        pass

    def startMatch(self):
        pass

    def bestMove(self, board: shogi.Board, legal_moves: 'set[shogi.Move]'):
        # 手がない場合は負け
        if len(legal_moves) == 0:
            return None
        # ランダムに手を返す
        else:
            return list(legal_moves)[random.randint(0, len(legal_moves)) - 1]


class Reason(Enum):
    CHECK_MATE = 'CHECK_MATE'
    TORYO = 'TORYO'
    ILLEGAL_MOVE = 'ILLEGAL_MOVE'
    SENNICHITE = 'SENNICHITE'
    MAX_MOVE = 'MAX_MOVE'
    # TODO TIMEOUT = 'TIMEOUT'
    # TPDP TRU = 'TRY'
    # TODO ABNORMAL = 'ABNORMAL'


class Result(Enum):
    BLACK_WIN = 'BLACK_WIN'
    WHITE_WIN = 'WHITE_WIN'
    DRAW = 'DRAW'


class GameResult():

    def __init__(self, black: Player, white: Player, startTime: dt = None):
        self.black_player_name = black.name
        self.white_player_name = white.name
        self.reason = None
        self.result = None
        self.move_num = None
        self.startTime = dt.now() if startTime is None else startTime
        self.endTime = None

    def setResult(self, result: Result, reason: Reason, move_num: int, endTime: dt = None):
        self.reason = reason
        self.result = result
        self.move_num = move_num
        self.endTime = dt.now() if endTime is None else endTime


class Game():
    @ staticmethod
    def opposit(turn: int):
        return shogi.WHITE if turn == shogi.BLACK else shogi.BLACK

    @ classmethod
    def endResult(cls, turn: int):
        return Result.BLACK_WIN if turn == shogi.BLACK else Result.WHITE_WIN

    def __init__(self, black=Player(), white=Player()):
        self.players = [black, white]
        self.board = shogi.Board()
        self.display_kifu = True
        self.game_result = None
        self.max_move = 256

    def setDisplayKif(self, display):
        self.display_kifu = display

    def setPlayer(self, color: int, player: Player):
        self.players[color] = player

    def setPlayerConfig(self, color: int, key, value):
        self.players[color].setConfig(key, value)

    def prepare(self):
        for player in self.players:
            player.prepare()

    def playMatch(self) -> GameResult:
        self.board.reset()
        self.game_result = GameResult(
            self.players[0], self.players[1])
        for player in self.players:
            player.startMatch()

        while not self.processOneMove():
            pass

        return self.game_result

    def processOneMove(self):
        if self.display_kifu:
            print(f'{self.board.move_number}手目')
        end = True

        turn = self.board.turn
        opposit = self.opposit(turn)
        player = self.players[self.board.turn]
        legal_moves = {move for move in self.board.legal_moves}

        move = player.bestMove(copy.deepcopy(self.board), legal_moves)

        # 投了の場合
        if move is None:
            # 投了
            self.game_result.setResult(self.endResult(
                opposit), Reason.TORYO, self.board.move_number)
            if self.display_kifu:
                print(f'{"先手" if turn == shogi.BLACK else "後手"}投了')

        # NullMoveチェック/合法手チェック
        elif not (move and move in legal_moves):
            self.game_result.setResult(self.endResult(
                opposit), Reason.ILLEGAL_MOVE, self.board.move_number)
            if self.display_kifu:
                print(f'{"先手" if turn == shogi.BLACK else "後手"}反則')
        else:
            if self.display_kifu:
                print(moveToStr(move, self.board))

            # 手を進める
            self.board.push(move)

            if self.display_kifu:
                print(self.board.kif_str())

            # 千日手チェック
            # TODO 王手千日手チェック
            if self.board.is_fourfold_repetition():
                self.game_result.setResult(
                    Result.DRAW, Reason.SENNICHITE, self.board.move_number)
                if self.display_kifu:
                    print('千日手')
            # 最大手数チェック
            elif self.board.move_number == self.max_move:
                self.game_result.setResult(
                    Result.DRAW, Reason.MAX_MOVE, self.board.move_number)
                if self.display_kifu:
                    print('最大手数')

            else:
                # 通常ルート
                end = False

        return end


def moveToStr(move: shogi.Move, board: shogi.Board):
    turn = '▲' if board.turn == shogi.BLACK else '△'
    to_sq = f'{9 - move.to_square % 9}{move.to_square // 9 + 1}'
    if move.drop_piece_type:
        return f'{turn}{to_sq}{shogi.PIECE_JAPANESE_SYMBOLS[move.drop_piece_type]}打ち'
    else:
        from_sq = f'{9 - move.from_square % 9}{move.from_square // 9 + 1}'
        koma = shogi.PIECE_JAPANESE_SYMBOLS[board.piece_type_at(
            move.from_square)]
        promotion = '成' if move.promotion else ''
        return f'{turn}{to_sq}{koma}{promotion} ({from_sq})'


if __name__ == '__main__':
    match = Game()
    result = match.playMatch()

    print(f'black:{result.black_player_name}')
    print(f'white:{result.white_player_name}')
    print(f'{result.move_num}手')
    print(f'{result.result.value} {result.reason.value}')
