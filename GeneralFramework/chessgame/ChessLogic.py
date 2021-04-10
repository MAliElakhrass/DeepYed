import chess
import numpy as np

# CONSTANTS
NUMBER_SQUARES = 8


class Board:
    """
    Chess Board based on python-chess library
    """

    def __init__(self, n=None):
        self.n = n or NUMBER_SQUARES
        self.board = chess.Board()
        self.pieces = self.get_bitboard(self.board)

    @staticmethod
    def get_bitboard(board: chess.Board):
        x = np.zeros(NUMBER_SQUARES * NUMBER_SQUARES, dtype=np.int8)

        for square in range(NUMBER_SQUARES * NUMBER_SQUARES):
            piece: chess.Piece = board.piece_at(square)
            if piece:
                color = piece.color
                col = int(square % 8)
                row = int(square / 8)
                x[row * 8 + col] = -piece.piece_type if color == chess.BLACK else piece.piece_type

        return np.reshape(x, (8, 8))

    def make_move(self, move_uci: str):
        move = chess.Move.from_uci(move_uci)

        # Check promotion
        if self.board.piece_type_at(move.from_square) == chess.PAWN:
            if int(move.to_square / 8) in [1, 8]:
                move.promotion = chess.QUEEN # Comment je fais pour savoir si Queen ou cavalier?

        self.board.push(chess.Move.from_uci(str(move)))


if __name__ == '__main__':
    b = Board()
    print(b.pieces)
