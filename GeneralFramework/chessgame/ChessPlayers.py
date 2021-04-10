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
