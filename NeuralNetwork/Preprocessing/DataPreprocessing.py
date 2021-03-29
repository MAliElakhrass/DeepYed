import numpy as np
import chess.pgn
import re
import memory_profiler
import time
import random
import math

N_SQUARES = 64
N_SIDES = 2
N_PIECES = 6
N_EXTRA_BITS = 5
N_GAMES = 1228086
CHUNK_SIZE = 2500


class DataPreprocessing:
    def __init__(self):
        self.data = self.read_data()
        self.n_squares = N_SQUARES
        self.n_sides = N_SIDES
        self.n_pieces = N_PIECES
        self.n_extra_bits = N_EXTRA_BITS
        self.n_games = N_GAMES
        self.chunk_size = CHUNK_SIZE

        self.bitboards = []
        self.labels = []

        self.train_bitboards, self.valid_bitboards = [], []
        self.train_labels, self.valid_labels = [], []
        self.total_train, self.total_valid = 0, 0

    def read_data(self, path='../data/CCRL-4040.[1228086].pgn'):
        """
        Reads the data from a directory specified by a path
        :param path: The path to the data containing all the games
        :return: A file object
        """
        self.n_games = int(re.search(r"\[([A-Za-z0-9_]+)\]", path).group(1))
        return open(path)

    def get_bitboard(self, board):
        """
        This method transforms the board to a binary bit-string of size 773 called a bitboard representation.

        :param board: The board
        :return: The bitboard representation of the position
        The dimension is (64 x 6 x 2) + 5.
        64: A chess board is 8 by 8, therefore there are 64 squares.
        6: There are 6 types of pieces (pawn, knight, bishop, rook, queen, and king)
        2: There are two sides in chess (White and Black)
        5: Additional 5 bits for castling rights,
            Fifth bit: The side to move (1 for White and 0 for Black)
            Fourth bit: If White can do a kingside castling right
            Third bit: If Black can do a kingside castling right
            Second bit If White can do a queenside castling right
            First bit: If Black can do a queenside castling right

        :rtype: Numpy array of dimensions (64 x 6 x 2) + 5
        """

        bitboard = np.zeros(self.n_squares * self.n_pieces * self.n_sides + self.n_extra_bits)

        for square in range(self.n_squares):
            current_piece = board.piece_at(square)

            if current_piece:
                # board.color_at() returns True if White, False if Black
                # Add +1, therefore if White, 1 will be placed at an even position or 1 will be placed at odd position
                current_color = int(board.color_at(square)) + 1
                bitboard[((current_piece.piece_type - 1) + square * 6) * current_color] = 1

        bitboard = self.fill_extra_bits(bitboard, board)
        return bitboard

    def fill_extra_bits(self, bitboard, board):
        """
        Fills the extra 5 bits with the corresponding value based on the trun and the castling rights

        :param bitboard: The current bitboard which the last 5 bits will be updated
        :param board:  The current board
        :return: The updated bitboard with the 5 last bits updated with the right values
        """
        bitboard[-1] = int(board.turn)
        bitboard[-2] = int(board.has_kingside_castling_rights(True))
        bitboard[-3] = int(board.has_kingside_castling_rights(False))
        bitboard[-4] = int(board.has_queenside_castling_rights(True))
        bitboard[-5] = int(board.has_queenside_castling_rights(False))

        return bitboard

    def retrieve_label(self, game):
        """
        Retrieves the outcome value of the current game

        :param game: The current parsed game
        :return: An integer 1 if White won, -1 if Black won and 0 if it's a draw
        """
        # Returns a string of format '0-1'
        label = game.headers['Result'].split('-')
        if label[0] == '1':
            return 1
        elif label[0] == '0':
            return -1
        else:
            return 0

    def update_per_move(self, board, move):
        """
        Adds a move to the current board and gets the bitboard of the board with the new move taken in consideration

        :param board: The current board
        :param move: The new move to add
        :return: board: The updated board with the new move,
        bitboard: The bitboard with the new move taken in consideration
        """
        board.push(move)
        bitboard = self.get_bitboard(board)
        return board, bitboard

    def save_results(self, directory="./PreprocessedData/"):
        """
        Saves the bitboards and the labels as numpy files

        :param directory: The directory path where the numpy arrays will be saved

        """
        # Random shuffle the data
        shuffled_list = list(zip(self.bitboards, self.labels))
        random.shuffle(shuffled_list)

        self.bitboards, self.labels = zip(*shuffled_list)
        self.train_bitboards = self.bitboards[:int(len(self.bitboards) * 0.8)]
        self.valid_bitboards = self.bitboards[int(len(self.bitboards) * .8):]
        self.train_labels = self.labels[:int(len(self.bitboards) * 0.8)]
        self.valid_labels = self.labels[int(len(self.bitboards) * .8):]

        np_bitboards = np.array(list(self.train_bitboards))
        np_labels = np.array(list(self.train_labels))

        train_name = str(self.total_train)

        np.save(directory + "Bitboards/Train/" + train_name, np_bitboards)
        np.save(directory + "Labels/Train/" + train_name, np_labels)

        val_bitboards = np.array(list(self.valid_bitboards))
        val_labels = np.array(list(self.valid_labels))

        val_name = str(self.total_valid)

        np.save(directory + "Bitboards/Valid/" + val_name, val_bitboards)
        np.save(directory + "Labels/Valid/" + val_name, val_labels)

    def fill_all_moves_data(self, game):
        """
        Adds the bitboard after every move of a game and the outcome of the game
        :param game: The current parsed game
        """
        # An iterable over the main moves after the current game
        all_moves = game.mainline_moves()

        # Get the outcome of the game
        game_outcome = self.retrieve_label(game)
        board = game.board()

        # Generate bitboard for every move and append the results
        for move in all_moves:
            board, bitboard = self.update_per_move(board, move)
            self.bitboards.append(bitboard)
            self.labels.append(game_outcome)

    def preprocess_games(self):
        """
        Preprocesses the games to extract he bitboards and the outcomes of every game
        """
        num_games = 0
        chunk_size = 0
        for _ in range(self.n_games):
            if num_games % 10 == 0:
                print(num_games)
                print("Train bitboards: " + str(self.total_train))
                print("Valid bitboards: " + str(self.total_valid))
            read_game = chess.pgn.read_game(self.data)
            game_outcome = self.retrieve_label(read_game)
            if game_outcome != 0:
                num_games += 1
                chunk_size += 1
                self.fill_all_moves_data(game=read_game)
                # Saving after chunk_size games with the name of the file being the number of bitboards where we're at
                if chunk_size == self.chunk_size or num_games == (self.n_games - (self.n_games % self.chunk_size)):
                    chunk_size = 0
                    self.total_train += math.floor(len(self.bitboards) * 0.8)
                    self.total_valid += math.ceil(len(self.bitboards) * 0.2)
                    self.save_results()
                    self.bitboards = []
                    self.labels = []


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing()
    m1 = memory_profiler.memory_usage()
    t1 = time.time()
    data_preprocessing.preprocess_games()
    t2 = time.time()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} Secs and {mem_diff} Mb to execute the method")