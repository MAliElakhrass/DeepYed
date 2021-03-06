import chess.pgn
import numpy as np
import random


# CONSTANTS
MAX_MOVES = 3000000


class DataGenerator:
    def __init__(self, data_path='NeuralNetKeras/data/games_data.pgn'):
        self.white_moves = np.zeros((MAX_MOVES, 2 * 6 * 64 + 5), dtype=np.int8)
        self.black_moves = np.zeros((MAX_MOVES, 2 * 6 * 64 + 5), dtype=np.int8)
        self.game = None
        self.data_path = data_path

    def get_valid_moves(self):
        """
        This function will read the moves made from the game.
        It will only save the move if it's not a capture move or if it's played after the 5th move

        :return:
        """
        valid_moves = []
        for i, move in enumerate(self.game.mainline_moves()):
            if not self.game.board().is_capture(move) and i >= 5:
                valid_moves.append(move)

        return valid_moves

    def get_bitboard(self, board):
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
                bitboard[(6*color + piece_indices[board.piece_at(i).symbol().lower()] + 12*i)] = 1

        bitboard[-1] = int(board.turn)
        bitboard[-2] = int(board.has_kingside_castling_rights(True))
        bitboard[-3] = int(board.has_kingside_castling_rights(False))
        bitboard[-4] = int(board.has_queenside_castling_rights(True))
        bitboard[-5] = int(board.has_queenside_castling_rights(False))

        return bitboard

    def add_move(self, index, white):
        """
        This function will add 10 random moves for a game

        :param white:
        :param index:
        :return: new index
        """
        valid_moves = self.get_valid_moves()

        # For 10 random moves
        selected_moves = random.sample(valid_moves, 10)

        board = chess.Board()
        for i, move in enumerate(self.game.mainline_moves()):
            board.push(move)

            if index >= MAX_MOVES:
                break

            if move in selected_moves:
                if white:
                    self.white_moves[index] = self.get_bitboard(board)
                else:
                    self.black_moves[index] = self.get_bitboard(board)

                index += 1

        return index

    def iterate_over_data(self):
        """
        This function iterates over the pgn file and extracts 10 random moves of each game

        :return:
        """
        data_file = open(self.data_path)
        black_moves_count, white_moves_count, count = 0, 0, 0
        while True:
            if count % 1000 == 0:
                print("Game Number: {count}\twhite moves: {white_moves}\tblack moves: {black_moves}".format(
                    count=count,
                    black_moves=black_moves_count,
                    white_moves=white_moves_count))

            self.game = chess.pgn.read_game(data_file)
            if not self.game or (white_moves_count >= MAX_MOVES and black_moves_count >= MAX_MOVES):
                break

            if self.game.headers["Result"] == "1-0" and white_moves_count < MAX_MOVES:
                white_moves_count = self.add_move(white_moves_count, white=True)
            if self.game.headers["Result"] == "0-1" and black_moves_count < MAX_MOVES:
                black_moves_count = self.add_move(black_moves_count, white=False)

            count += 1

    def save(self):
        """
        This function will save the data into two files

        :return:
        """
        np.save('NeuralNetKeras/data/white.npy', self.white_moves)
        np.save('NeuralNetKeras/data/black.npy', self.black_moves)


if __name__ == '__main__':
    data_generator = DataGenerator()
    data_generator.iterate_over_data()
    data_generator.save()
