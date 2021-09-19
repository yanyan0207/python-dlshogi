import shogi
from game import GameResult, Reason, Result

KOMA_LIST = ['', 'FU', 'KY', 'KE', 'GI', 'KI', 'KA', 'HI', 'OU',
                 'TO', 'NY', 'NK', 'NG',       'UM', 'RY']


def createKif(board: shogi.Board, ofile: str, gameResult: GameResult):
    lines = ['V2',
             f'N+{gameResult.black_player_name}',
             f'N-{gameResult.white_player_name}',
             f'$START_TIME:{gameResult.startTime.strftime("%Y/%m/%d %H:%M:%S")}']

    lines += ['P1-KY-KE-GI-KI-OU-KI-GI-KE-KY']
    lines += ['P2 * -HI *  *  *  *  * -KA * ']
    lines += ['P3-FU-FU-FU-FU-FU-FU-FU-FU-FU']
    lines += ['P4 *  *  *  *  *  *  *  *  * ']
    lines += ['P5 *  *  *  *  *  *  *  *  * ']
    lines += ['P6 *  *  *  *  *  *  *  *  * ']
    lines += ['P7+FU+FU+FU+FU+FU+FU+FU+FU+FU']
    lines += ['P8 * +KA *  *  *  *  * +HI * ']
    lines += ['P9+KY+KE+GI+KI+OU+KI+GI+KE+KY']

    tmp_board = shogi.Board()
    for move in board.move_stack:
        turn = '+' if tmp_board.turn == shogi.BLACK else '-'
        from_csa = '00' if move.from_square is None else f'{9 - move.from_square % 9}{move.from_square // 9 + 1}'
        to_csa = f'{9 - move.to_square % 9}{move.to_square // 9 + 1}'
        tmp_board.push(move)
        piece = tmp_board.piece_type_at(move.to_square)
        lines += [f'{turn}{from_csa}{to_csa}{KOMA_LIST[piece]}']
        lines += ['T0']

    if gameResult.reason == Reason.CHECK_MATE:
        lines += ['%TSUMI']
        reason = ""
    elif gameResult.reason == Reason.TORYO:
        lines += ['%TORYO']
        reason = 'toryo'
    elif gameResult.reason == Reason.ILLEGAL_MOVE:
        lines += ['%ILLEGAL_MOVE']
        reason = 'illegal move'
    elif gameResult.reason == Reason.SENNICHITE:
        lines += ['%SENNICHITE']
        reason = 'sennichite'
    elif gameResult.reason == Reason.MAX_MOVE:
        reason = 'max_moves'

    if gameResult.result == Result.BLACK_WIN:
        first = f'{gameResult.black_player_name} win'
        second = f'{gameResult.white_player_name} lose'
    elif gameResult.result == Result.WHITE_WIN:
        first = f'{gameResult.white_player_name} win'
        second = f'{gameResult.black_player_name} lose'
    else:
        first = f'{gameResult.black_player_name} draw'
        second = f'{gameResult.white_player_name} draw'

    lines += [f"'summary:{reason}:{first}:{second}"]
    lines = [line + '\n' for line in lines]
    with open(ofile, mode='w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    import game
    match = game.Game()
    result = match.playMatch()
    createKif(match.board, 'test.csa', result)
