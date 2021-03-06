import chess
import numpy as np

# CONSTANTS
NUMBER_SQUARES = 8


class Board:
    """
    Chess Board based on python-chess library
    """

    def __init__(self, board: chess.Board = None):
        self.n = NUMBER_SQUARES
        self.board = board or chess.Board()

        self.letters = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
        self.numbers = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}

    @staticmethod
    def get_bitboard(board: chess.Board):
        x = np.zeros(64, dtype=np.int8)
        # print('Flipping: ', flip)
        for pos in range(64):
            piece = board.piece_type_at(pos)  # Gets the piece type at the given square. 0==>blank,1,2,3,4,5,6
            if piece:
                color = int(
                    bool(board.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))  # to check if piece is black or white
                col = int(pos % 8)
                row = int(pos / 8)
                x[row * 8 + col] = -piece if color else piece

        return np.reshape(x, (8, 8))

    def make_move(self, move_uci: str):
        move = chess.Move.from_uci(move_uci)

        # Check promotion
        if self.board.piece_type_at(move.from_square) == chess.PAWN:
            if int(move.to_square / 8) in [1, 8]:
                move.promotion = chess.QUEEN # Comment je fais pour savoir si Queen ou cavalier?

        try:
            self.board.push(chess.Move.from_uci(str(move)))
        except:
            print("FAKE MOVE")
        # self.pieces = self.get_bitboard(self.board)

    def get_valid_moves(self):
        legal_moves = list(self.board.legal_moves)
        for i, move in enumerate(legal_moves):
            move = str(move)
            legal_moves[i] = [self.letters[move[0]] - 1, self.numbers[move[1]] - 1, self.letters[move[2]] - 1,
                              self.numbers[move[3]] - 1]

        return legal_moves

    def count_diff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        pieces = self.get_bitboard(self.board)
        whites = np.count_nonzero(pieces < 0)
        blacks = np.count_nonzero(pieces > 1)

        return whites - blacks


if __name__ == '__main__':
    b = Board()
    print(b.get_valid_moves())
