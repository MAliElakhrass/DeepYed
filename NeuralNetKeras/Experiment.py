import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from datetime import datetime
from stockfish import Stockfish
from tensorflow.keras.models import load_model
import chess
import chess.pgn
import chess.polyglot
import numpy as np
import sys


class Experiment:
    def __init__(self):
        self.deepyed = load_model('NeuralNetKeras/model/DeepYed.h5')

    def play_against_stockfish(self, level=1, depth=3, number_games=10):
        stockfish = Stockfish('engines/stockfish.exe',
                              parameters={"Threads": 4, "Skill Level": level})

        for i in range(number_games):
            game = chess.pgn.Game()
            game.headers["Event"] = "Evalutation DeepYed vs Stockfish (Neural Net)"
            game.headers["Site"] = "My PC"
            game.headers["Date"] = str(datetime.now().date())
            game.headers["Round"] = str(i+1)
            game.headers["White"] = "DeepYed"
            game.headers["Black"] = 'Stockfish'

            board = chess.Board()
            history = []
            while not board.is_game_over():
                if board.turn:
                    print("DeepYed's Turn")
                    _, move = self.alphabeta(board, depth, -100, 100, True, board)
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

            print(game, file=open(f"NeuralNetKeras/level_{level}_round_{i+1}.pgn", "w"), end="\n\n")

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
        bitboard_1, bitboard_2 = self.get_bitboard(move_1), self.get_bitboard(move_2)
        return self.apply_siamese_network(bitboard_1, bitboard_2)[0][0]

    def alphabeta(self, board, depth, alpha, beta, white, orig_board):
        try:
            move = chess.polyglot.MemoryMappedReader('./books/Perfect2017-SF12.bin')\
                .weighted_choice(board=orig_board).move
            return None, move
        except:
            if depth == 0:
                yed = self.compare_moves(board, orig_board)
                return yed, None
            if white:
                v = -100
                moves = list(board.generate_legal_moves())
                best_move = None
                for move in moves:
                    new_board = board.copy()
                    new_board.push(move)
                    candidate_v, _ = self.alphabeta(new_board, depth - 1, alpha, beta, False, orig_board)
                    if candidate_v >= v:
                        v = candidate_v
                        best_move = move
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break
                return v, best_move
            else:
                v = 100
                moves = list(board.generate_legal_moves())
                best_move = None
                for move in moves:
                    new_board = board.copy()
                    new_board.push(move)
                    candidate_v, _ = self.alphabeta(new_board, depth - 1, alpha, beta, True, orig_board)
                    if candidate_v <= v:
                        v = candidate_v
                        best_move = move
                    else:
                        pass
                    beta = min(beta, v)
                    if beta <= alpha:
                        break
                return v, best_move

    @staticmethod
    def get_move_stockfish(board, stockfish):
        fen = board.fen()
        stockfish.set_fen_position(fen)

        return stockfish.get_best_move_time(100)


if __name__ == '__main__':
    level = int(sys.argv[1]) or 1
    depth = int(sys.argv[2]) or 3
    number_games = int(sys.argv[3]) or 10

    experiment = Experiment()
    experiment.play_against_stockfish(level=level, depth=depth, number_games=number_games)
