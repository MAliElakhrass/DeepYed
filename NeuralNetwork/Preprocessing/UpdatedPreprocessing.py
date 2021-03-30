import numpy as np
import chess.pgn
import re
import memory_profiler
import time
import random
import math
import chess

N_SQUARES = 64
N_SIDES = 2
N_PIECES = 6
N_EXTRA_BITS = 5
N_GAMES = 1228086
CHUNK_SIZE = 500000


class UpdatedPreprocessing:
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

        self.train_bitboards, self.valid_bitboards, self.test_bitboards = [], [], []
        self.train_labels, self.valid_labels, self.test_labels = [], [], []
        self.total_train, self.total_valid, self.total_test = 0, 0, 0

    def read_data(self, path='./data/CCRL-4040.[1228086].pgn'):
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

        bitboard = np.zeros(self.n_squares * self.n_pieces * self.n_sides + self.n_extra_bits, dtype=np.int8)

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

    def retrieve_ELO(self, game):
        """
        Retrieves the ELO of Wite and Black for the current game

        :param game: The current parsed game
        :return: White ELO and Black ELO
        """
        white_Elo = int(game.headers['WhiteElo'])
        black_Elo = int(game.headers['BlackElo'])

        return white_Elo, black_Elo

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

    def save_results(self, directory="NeuralNetwork/PreprocessedData/"):
        """
        Saves the bitboards and the labels as numpy files

        :param directory: The directory path where the numpy arrays will be saved

        """
        # Random shuffle the data
        shuffled_list = list(zip(self.bitboards, self.labels))
        random.shuffle(shuffled_list)
        # Split data 80, 10, 10
        self.bitboards, self.labels = zip(*shuffled_list)
        self.train_bitboards = self.bitboards[:int(len(self.bitboards) * 0.8)]
        self.valid_bitboards = self.bitboards[int(len(self.bitboards) * .8):int(len(self.bitboards) * .9)]
        self.test_bitboards = self.bitboards[int(len(self.bitboards) * .9):]

        self.train_labels = self.labels[:int(len(self.bitboards) * 0.8)]
        self.valid_labels = self.labels[int(len(self.bitboards) * .8):int(len(self.bitboards) * .9)]
        self.test_labels = self.labels[int(len(self.bitboards) * .9):]

        self.total_train += len(self.train_bitboards)
        self.total_valid += len(self.valid_bitboards)
        self.total_test += len(self.test_bitboards)

        np_bitboards = np.array(list(self.train_bitboards), dtype=np.int8)
        np_labels = np.array(list(self.train_labels), dtype=np.int8)

        train_name = str(self.total_train)

        np.save(directory + "Bitboards/Train/" + train_name, np_bitboards)
        np.save(directory + "Labels/Train/" + train_name, np_labels)

        val_bitboards = np.array(list(self.valid_bitboards), dtype=np.int8)
        val_labels = np.array(list(self.valid_labels), dtype=np.int8)

        val_name = str(self.total_valid)

        np.save(directory + "Bitboards/Valid/" + val_name, val_bitboards)
        np.save(directory + "Labels/Valid/" + val_name, val_labels)

        test_bitboards = np.array(list(self.test_bitboards), dtype=np.int8)
        test_labels = np.array(list(self.test_labels), dtype=np.int8)

        test_name = str(self.total_test)

        np.save(directory + "Bitboards/Test/" + test_name, test_bitboards)
        np.save(directory + "Labels/Test/" + test_name, test_labels)

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
        n_moves = 0

        valid_bitboards = []
        # Generate bitboard for every move and append the results
        for move in all_moves:
            n_moves += 1
            detailed_move = board.san(move)
            board, bitboard = self.update_per_move(board, move)
            # Ignore the first 5 moves and ignore capture moves
            if n_moves > 5 and "x" not in detailed_move:
                valid_bitboards.append(bitboard)

        # Pick 10 random valid moves from the game
        random_set_bitboards = random.sample(valid_bitboards, 10)
        random_set_labels = [game_outcome] * len(random_set_bitboards)
        self.bitboards.extend(random_set_bitboards)
        self.labels.extend(random_set_labels)

    def preprocess_games(self):
        """
        Preprocesses the games to extract he bitboards and the outcomes of every game
        """
        num_games = 0
        total_games = 0
        for _ in range(self.n_games):
            total_games += 1
            if num_games % 10 == 0:
                print("Total choosen Games: " + str(num_games))
                print("Total games: " + str(total_games))
                print("Length Bitboards: " + str(len(self.bitboards)))
                print("Train bitboards: " + str(self.total_train))
                print("Valid bitboards: " + str(self.total_valid))
            read_game = chess.pgn.read_game(self.data)
            game_outcome = self.retrieve_label(read_game)
            white_ELO, black_ELO = self.retrieve_ELO(read_game)
            if game_outcome != 0 and white_ELO > 2000 and black_ELO > 2000:
                num_games += 1

                self.fill_all_moves_data(game=read_game)
                # Saving after chunk_size games with the name of the file being the number of bitboards where we're at
                if len(self.bitboards) >= self.chunk_size or num_games == self.n_games - 1:
                    chunk_size = 0
                    self.save_results()
                    self.bitboards = []
                    self.labels = []


if __name__ == "__main__":
    data_preprocessing = UpdatedPreprocessing()
    m1 = memory_profiler.memory_usage()
    t1 = time.time()
    data_preprocessing.preprocess_games()
    t2 = time.time()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} Secs and {mem_diff} Mb to execute the method with dtype uint8")
