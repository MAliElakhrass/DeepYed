import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from datetime import datetime
from stockfish import Stockfish
from tensorflow.keras.models import load_model
import chess
import chess.pgn
import chess.polyglot
import numpy as np


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
                _, move = self.alphabeta(board, 3, -100, 100, True, board)
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
        bitboard_1, bitboard_2 = self.get_bitboard(move_1), self.get_bitboard(move_2)
        return self.apply_siamese_network(bitboard_1, bitboard_2)[0][0]

    def alphabeta(self, board, depth, alpha, beta, white, orig_board):
        try:
            move = chess.polyglot.MemoryMappedReader('./books/Perfect2017-SF12.bin').weighted_choice(
                board=orig_board).move
            return None, move
        except:
            if depth == 0:
                yed = self.compare_moves(board, orig_board)
                return yed, None
            if white:
                v = -100  # very (relatively) small number
                moves = board.generate_legal_moves()
                moves = list(moves)
                best_move = None
                for move in moves:
                    new_board = board.copy()
                    new_board.push(move)
                    candidate_v, _ = self.alphabeta(new_board, depth - 1, alpha, beta, True, orig_board)
                    if candidate_v >= v:
                        v = candidate_v
                        best_move = move
                    else:
                        pass
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break
                return v, best_move
            else:
                v = 100  # very (relatively) large number
                moves = board.generate_legal_moves()
                moves = list(moves)
                best_move = None
                for move in moves:
                    new_board = board.copy()
                    new_board.push(move)
                    candidate_v, _ = self.alphabeta(new_board, depth - 1, alpha, beta, False, orig_board)
                    if candidate_v <= v:
                        v = candidate_v
                        best_move = move
                    else:
                        pass
                    beta = min(beta, v)
                    if beta <= alpha:
                        break
                return v, best_move

    def get_move_stockfish(self, board, stockfish):
        fen = board.fen()
        stockfish.set_fen_position(fen)

        return stockfish.get_best_move_time(100)


if __name__ == '__main__':
    experiment = Experiment()
    experiment.play_against_stockfish()
