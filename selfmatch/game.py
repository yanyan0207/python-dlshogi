import shogi
import copy
import random
from enum import Enum
from shogi.Consts import BLACK, WHITE


class Player():
    def setConfig(self, key, value):
        pass

    def startMatch(self):
        pass

    def bestMove(self, board: shogi.Board):
        # 合法手の数を算出
        legal_moves = [move for move in board.legal_moves]

        # 手がない場合は負け

        if len(legal_moves) == 0:
            return None

        # ランダムに手を返す
        else:
            return legal_moves[random.randint(0, len(legal_moves)) - 1]


class Game():
    class EndReason(Enum):
        TORYO = "TORYO"
        ILLEGAL_MOVE = "ILLEGAL_MOVE"
        SENNICHITE = "SENNICHITE"
        TSUMI = "TSUMI"
        # TODO TIMEOUT = "TIMEOUT"
        # TPDP TRU = "TRY"

    class EndResult(Enum):
        BLACK_WIN = "BLACK_WIN"
        WHITE_WIN = "WHITE_WIN"
        DRAW = "DRAW"

    @staticmethod
    def opposit(turn: int):
        return shogi.WHITE if turn == shogi.BLACK else shogi.BLACK

    @classmethod
    def endResult(cls, turn: int):
        return turn == cls.EndResult.BLACK_WIN if turn == shogi.BLACK else cls.EndResult.WHITE_WIN

    def __init__(self):
        self.players = [Player(), Player()]
        self.board = shogi.Board()

    def setPlayer(self, color: int, player: Player):
        self.players[color] = player

    def setConfig(self, color: int, key, value):
        self.players[color].setConfig(key, value)

    def startMatch(self):
        self.board.reset()
        for player in self.players:
            player.startMatch()

    def processOneMove(self):
        print(f'{self.board.move_number}手目')
        end = True

        turn = self.board.turn
        opposit = self.opposit(turn)
        player = self.players[self.board.turn]
        move = player.bestMove(copy.deepcopy(self.board))

        # 投了の場合
        if move is None:
            # 投了
            self.win = self.endResult(opposit)
            self.end_reason = Game.EndReason.TORYO
            print(f'{"先手" if turn == shogi.BLACK else "後手"}投了')

        # NullMoveチェック/合法手チェック
        elif not (move and move in self.board.pseudo_legal_moves):
            self.win = self.endResult(opposit)
            self.end_reason = Game.EndReason.ILLEGAL_MOVE
            print(f'{"先手" if turn == shogi.BLACK else "後手"}反則')
        else:
            print(moveToStr(move, self.board))
            self.board.push(move)
            print(self.board.kif_str())
            # 千日手チェック
            # TODO 王手千日手チェック
            if self.board.is_fourfold_repetition():
                self.win = self.EndResult.DRAW
                self.end_reason = Game.EndReason.SENNICHITE
                print('千日手')
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


if __name__ == "__main__":
    game = Game()
    while not game.processOneMove():
        pass
    print(game.win, game.end_reason)
