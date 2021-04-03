import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from datetime import datetime
from stockfish import Stockfish
from tensorflow.keras.models import load_model
import chess
import chess.pgn
import chess.polyglot
import numpy as np
import random


class Experiment:
    def __init__(self):
        self.deepyed = load_model('./model/DeepYed.h5')

    def play_against_stockfish(self):
        stockfish = Stockfish('./engines/stockfish-12/stockfish.exe',
                              parameters={"Threads": 4, "Skill Level": 1})

        game = chess.pgn.Game()
        game.headers["Event"] = "DeepYed vs Stockfish"
        game.headers["Site"] = "Hammad's PC"
        game.headers["Date"] = str(datetime.now().date())
        game.headers["Round"] = '1'
        game.headers["White"] = "DeepYed"
        game.headers["Black"] = 'Stockfish'

        board = chess.Board()
        history = []
        while not board.is_game_over():
            if board.turn:
                print("DeepYed's Turn")
                try:
                    move = chess.polyglot.MemoryMappedReader('./books/Perfect2017-SF12.bin').weighted_choice(
                        board=board).move
                except:
                    _, move = self.alpha_beta_max(board, None, None, None, None, 3)
                history.append(move)
                board.push(move)
                print(move)
            else:
                print("Stockfish's Turn")
                move = chess.Move.from_uci(self.get_move_stockfish(board, stockfish))
                history.append(move)
                board.push(move)
                print(move)

        game.add_line(history)
        game.headers["Result"] = str(board.result())

        print(game)
        print(game, file=open("test.pgn", "w"), end="\n\n")

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

    def apply_siamese_network(self, bitboard_1, bitboard_2):
        bitboard_1 = np.expand_dims(bitboard_1, axis=0)
        bitboard_2 = np.expand_dims(bitboard_2, axis=0)
        return self.deepyed([bitboard_1, bitboard_2])

    def compare_moves(self, move_1, move_2):
        # bitboard_1, bitboard_2 = self.get_bitboard(move_1), self.get_bitboard(move_2)
        value = self.apply_siamese_network(move_1, move_2)
        return value[0]

    def alpha_beta_max(self, board, alpha_pos, alpha_move, beta_pos, beta_move, depth):
        if depth == 0 or board.is_game_over():
            return self.get_bitboard(board)

        # Copy the board
        child_board = board.copy()

        # Randomize legal moves
        legal_moves = random.sample(list(board.legal_moves), len(list(board.legal_moves)))

        # For every legal move,
        for move in legal_moves:

            # Make the move on the copy
            child_board.push(move)

            # Get the result of the move of the min player
            pos, _ = self.alpha_beta_min(child_board, alpha_pos, alpha_move, beta_pos, beta_move, depth - 1)

            # If the resulting position is better than the beta position, return the
            # beta position which is the least best position for the max player that
            # the min player is assured of. This represents fail-hard beta-cutoff.
            # Having a guaranteed worse outcome for the max player, the min player
            # won't enter this node since the max player has at least one move in
            # this node with an outcome that dominates beta position. Note: None
            # corresponds to an infinitely good outcome for the max player when
            # assigned to the beta position.
            if beta_pos is not None:
                comparison = np.squeeze(self.compare_moves(pos, beta_pos))
                if comparison[0] >= comparison[1]:
                    return beta_pos, beta_move

            # If the resulting position is better than the alpha position, assign it to
            # to the alpha position which is the least worst position for the max player
            # that the max player is assured of. After entering this node, the max player
            # can make at least this move that results in a better position than the alpha
            # position. Hence, the resulting position is the new guaranteed least worst case.
            # Note: None corresponds to an infinitely bad outcome for the max player when
            # assigned to the alpha position.
            if alpha_pos is None:
                alpha_pos = pos
                alpha_move = move
            else:
                comparison = np.squeeze(self.compare_moves(pos, alpha_pos))  # self.deep_chess.evaluate(self.sess, pos, alpha_pos))
                if comparison[0] > comparison[1]:
                    alpha_pos = pos
                    alpha_move = move

            # Unmake the move
            child_board = board.copy()

        # Return the alpha position (and the corresponding move) as
        # it is the guaranteed worst-case scenario for the max player.
        return alpha_pos, alpha_move

    def alpha_beta_min(self, board, alpha_pos, alpha_move, beta_pos, beta_move, depth):

        # Return the normalized bitboard representation when
        # the node is a final one or the depth limit is reached
        if depth == 0 or board.is_game_over():
            return self.get_bitboard(board), None

        # Copy the board
        child_board = board.copy()

        # Randomize legal moves
        legal_moves = random.sample(list(board.legal_moves), len(list(board.legal_moves)))

        # For every legal move,
        for move in legal_moves:

            # Make the move on the copy
            child_board.push(move)

            # Get the result of the move of the max player
            pos, _ = self.alpha_beta_max(child_board, alpha_pos, alpha_move, beta_pos, beta_move, depth - 1)

            # If the resulting position is worse than the alpha position, return the
            # alpha position which is the least worst position for the max player that
            # the max player is assured of. This represents fail-hard alpha-cutoff.
            # Having a guaranteed better outcome for the itself, the max player
            # won't enter this node since the min player has at least one move in
            # this node with an outcome that is worse than the alpha position.
            # Note: None corresponds to an infinitely bad outcome for the max player
            # when assigned to the alpha position.
            if alpha_pos is not None:
                comparison = np.squeeze(self.compare_moves(pos, alpha_pos))
                if comparison[0] <= comparison[1]:
                    return alpha_pos, alpha_move

            # If the resulting position is worse than the beta position, assign it to
            # to the beta position which is the least best position for the max player
            # that the min player is assured of. After entering this node, the min player
            # can make at least this move that results in a worse position for the max player
            # than the beta position. Hence, the resulting position is the new guaranteed
            # least worst case. Note: None corresponds to an infinitely good outcome for
            # the max player when assigned to the beta position.
            if beta_pos is None:
                beta_pos = pos
                beta_move = move
            else:
                comparison = np.squeeze(self.compare_moves(pos, beta_pos))
                if comparison[0] < comparison[1]:
                    beta_pos = pos
                    beta_move = move

            # Unmake the move
            child_board = board.copy()

        # Return the beta position (and the corresponding move) as
        # it is the guaranteed worst-case scenario for the min player.
        return beta_pos, beta_move

    def get_move_stockfish(self, board, stockfish):
        fen = board.fen()
        stockfish.set_fen_position(fen)

        return stockfish.get_best_move_time(100)


if __name__ == '__main__':
    experiment = Experiment()
    experiment.play_against_stockfish()
