from stockfish import Stockfish
import chess
import logging
import random

log = logging.getLogger(__name__)


class StockfishPlayer:
    def __init__(self, game, level):
        self.game = game
        self.engine = Stockfish('engines/stockfish', parameters={"Threads": 4, "Skill Level": level})

    def play(self, board):
        fen = board.fen()
        self.engine.set_fen_position(fen)

        move_uci = chess.Move.from_uci(self.engine.get_best_move_time(100))
        if move_uci not in list(board.legal_moves):
            log.error("Wrong move chosen by Stockfish! Check code")
            move_uci = random.choice(list(board.legal_moves))

        action = move_uci.from_square * 64 + move_uci.to_square

        return action
