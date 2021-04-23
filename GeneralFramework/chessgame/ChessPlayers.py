from stockfish import Stockfish
import chess
import numpy as np
import random


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)
        a = random.choice(np.nonzero(valid_moves))

        return a


class GreedyChessPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)
        candidates = []
        for i, move in enumerate(valid_moves):
            if move == 0:
                continue

            next_board, _ = self.game.getNextState(board, 1, i)
            score = self.game.get_score(board, 1)
            candidates += [(-score, i)]

        candidates.sort()

        return candidates[0][1]


class StockfishPlayer:
    def __init__(self, game, level):
        self.game = game
        self.engine = Stockfish('engines/stockfish-12/stockfish.exe', parameters={"Threads": 4, "Skill Level": level})
        self.letters = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}

    def play(self, board):
        fen = board.fen()
        print(fen)
        self.engine.set_fen_position(fen)

        move = self.engine.get_best_move_time(100)

        x1 = self.letters[move[0]]
        x2 = self.letters[move[2]]
        y1 = int(move[1])
        y2 = int(move[3])

        print((8 * 8) * (8 * x1 + y1) + (8 * x2 + y2)) + 1
        return (8 * 8) * (8 * x1 + y1) + (8 * x2 + y2)
