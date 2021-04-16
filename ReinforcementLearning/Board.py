import chess
import numpy as np


class Board:
    def __init__(self):
        self.current_board: chess.Board = chess.Board()
        self.current_board_bb = self.get_bitboard(self.current_board)
        self.move_count = 0
        self.player = 0  # current player's turn (0:white, 1:black)

    @staticmethod
    def get_bitboard(board):
        """
        Convert a board to numpy array of size 8x8
        :param board:
        :return: numpy array 8*8
        """
        bitboard = np.zeros([8, 8]).astype(str)

        for square in range(8 * 8):
            if board.piece_at(square):
                piece = board.piece_at(square).symbol()
            else:
                piece = ' '
            col = int(square % 8) - 8
            row = int(square / 8) - 8
            bitboard[row][col] = piece

        return bitboard

    def make_move(self, move):
        if move not in self.current_board.legal_moves:
            print('WARNING! Not a valid move')

        self.current_board.push(move)
        self.current_board_bb = self.get_bitboard(self.current_board)
        self.move_count += 1

    def check_winner(self):
        return self.current_board.is_game_over()

    def get_all_actions(self):
        return list(self.current_board.legal_moves)


if __name__ == '__main__':
    b = Board()
    print(b.current_board_bb)
