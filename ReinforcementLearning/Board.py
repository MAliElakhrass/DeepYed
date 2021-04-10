import chess
import numpy as np


class Board:
    def __init__(self):
        self.current_board: chess.Board = chess.Board()
        self.current_board_bb = self.get_bitboard(self.current_board)

    @staticmethod
    def get_bitboard(board):
        """
        Convert a board to bitboard of size 773
        :param board:
        :return:
        """
        bitboard = np.zeros(2 * 6 * 64 + 5, dtype=np.int8)

        piece_indices = {'p': 0,
                         'n': 1,
                         'b': 2,
                         'r': 3,
                         'q': 4,
                         'k': 5
                         }

        for i in range(8 * 8):
            if board.piece_at(i):
                color = int(board.piece_at(i).color)
                bitboard[(6 * color + piece_indices[board.piece_at(i).symbol().lower()] + 12 * i)] = 1

        bitboard[-1] = int(board.turn)
        bitboard[-2] = int(board.has_kingside_castling_rights(True))
        bitboard[-3] = int(board.has_kingside_castling_rights(False))
        bitboard[-4] = int(board.has_queenside_castling_rights(True))
        bitboard[-5] = int(board.has_queenside_castling_rights(False))

        return bitboard

    def make_move(self, move):
        if move not in self.current_board.legal_moves:
            print('WARNING! Not a valid move')

        self.current_board.push(move)
        self.current_board_bb = self.get_bitboard(self.current_board)

    def check_winner(self):
        return self.current_board.is_game_over()

    def get_all_actions(self):
        return list(self.current_board.legal_moves)
