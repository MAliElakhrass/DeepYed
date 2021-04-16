import chess
import numpy as np


class Board:
    def __init__(self):
        self.current_board: chess.Board = chess.Board()
        self.current_board_bb = self.get_bitboard(self.current_board)
        self.move_count = 0
        self.player = 0  # current player's turn (0:white, 1:black)
        self.en_passant = -999

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

        if self.current_board.is_en_passant(move):
            self.en_passant = move[-1] # VRAIMENT PAS SUR

    def check_winner(self):
        return self.current_board.is_game_over()

    def get_all_actions(self):
        return list(self.current_board.legal_moves)

    def encode_board(self):
        piece_dict = {"R": 0, "N": 1, "B": 2, "Q": 3, "K": 4, "P": 5, "r": 6, "n": 7, "b": 8, "q": 9, "k": 10, "p": 11}
        encoded = np.zeros([8, 8, 22], dtype=np.int8)

        for square in range(8 * 8):
            if self.current_board.piece_at(square):
                value = piece_dict[self.current_board.piece_at(square).symbol()]
                i = int(square / 8) - 8
                j = int(square % 8) - 8
                encoded[i, j, value] = 1

        if self.player == 1:
            encoded[:, :, 12] = 1

        # Castling for white
        encoded[:, :, 13] = int(not (self.current_board.has_queenside_castling_rights(color=chess.WHITE)))  # QUEENSIDE
        encoded[:, :, 14] = int(not (self.current_board.has_kingside_castling_rights(color=chess.WHITE)))  # KINGSIDE

        # Castling for blacks
        encoded[:, :, 15] = int(not (self.current_board.has_queenside_castling_rights(color=chess.BLACK)))  # QUEENSIDE
        encoded[:, :, 16] = int(not (self.current_board.has_kingside_castling_rights(color=chess.BLACK)))  # KINGSIDE

        encoded[:, :, 17] = self.move_count
        encoded[:, :, 21] = self.en_passant


if __name__ == '__main__':
    b = Board()
    print(b.current_board_bb)
