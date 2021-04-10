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

        self.letters = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
        self.numbers = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}

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

    def get_valid_moves(self):
        legal_moves = list(self.board.legal_moves)
        for i, move in enumerate(legal_moves):
            move = str(move)
            legal_moves[i] = [self.letters[move[0]] - 1, self.numbers[move[1]] - 1, self.letters[move[2]] - 1,
                              self.numbers[move[3]] - 1]

        return legal_moves


if __name__ == '__main__':
    b = Board()
    print(b.pieces)
    print(b.get_valid_moves())
    print(b.get_legal_moves())
