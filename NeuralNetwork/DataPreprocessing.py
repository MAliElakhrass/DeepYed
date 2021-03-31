from random import choice, randrange
import chess
import numpy as np


class DataPreprocessing:
    def __init__(self):
        self.squares_index = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
            'f': 5,
            'g': 6,
            'h': 7
        }

        self.boards = []

    def random_board(self, max_depth=200):
        board = chess.Board()
        depth = randrange(0, max_depth)

        for _ in range(depth):
            all_moves = list(board.legal_moves)
            random_move = choice(all_moves)
            board.push(random_move)
            if board.is_game_over():
                break

        return board

    def get_score(self, board, depth):
        with chess.engine.SimpleEngine.popen_uci('/content/stockfish') as sf:
            result = sf.analyse(board, chess.engine.Limit(depth=depth))
            score = result['score'].white().score()
            return score

    def square_to_index(self, square):
        letter = chess.square_name(square)
        return 8 - int(letter[1]), self.squares_index[letter[0]]

    def split_dims(self, board):
        # this is the 3d matrix
        board3d = np.zeros((14, 8, 8), dtype=np.int8)

        # here we add the pieces's view on the matrix
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, chess.WHITE):
                idx = np.unravel_index(square, (8, 8))
                board3d[piece - 1][7 - idx[0]][idx[1]] = 1
            for square in board.pieces(piece, chess.BLACK):
                idx = np.unravel_index(square, (8, 8))
                board3d[piece + 5][7 - idx[0]][idx[1]] = 1

        # add attacks and valid moves too
        # so the network knows what is being attacked
        aux = board.turn
        board.turn = chess.WHITE
        for move in board.legal_moves:
            i, j = self.square_to_index(move.to_square)
            board3d[12][i][j] = 1
        board.turn = chess.BLACK
        for move in board.legal_moves:
            i, j = self.square_to_index(move.to_square)
            board3d[13][i][j] = 1
        board.turn = aux

        return board3d
