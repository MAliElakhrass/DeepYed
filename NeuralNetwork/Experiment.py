from Architecture.ImprovedAutoEncoder import ImprovedAutoEncoder
from Architecture.ImprovedSiamese import ImprovedSiamese
from Preprocessing.AprilPreprocessing import AprilPreprocessing
from datetime import datetime
from stockfish import Stockfish
import torch
import numpy as np
import chess
import chess.pgn
import chess.polyglot


N_SQUARES = 64
N_SIDES = 2
N_PIECES = 6
N_EXTRA_BITS = 5
N_GAMES = 1228080
CHUNK_SIZE = 2000000


class Experiment():
    def __init__(self, DBN_model_path, siamese_model_path):
        self.DBN_model_path = DBN_model_path
        self.siamese_model_path = siamese_model_path
        device = torch.device("cpu")
        self.autoencoder = ImprovedAutoEncoder().to(device)
        self.siamese_net = ImprovedSiamese().to(device)
        self.load_model_states()
        self.autoencoder.eval()
        self.siamese_net.eval()

    def load_model_states(self):
        autoencoder_state_dict = torch.load(self.DBN_model_path, map_location=lambda storage, loc: storage)
        siamese_state_dict = torch.load(self.siamese_model_path, map_location=lambda storage, loc: storage)

        self.autoencoder.load_state_dict(autoencoder_state_dict['state_dict'])
        self.siamese_net.load_state_dict(siamese_state_dict['state_dict'])

    def extract_features(self, bitboards):
        tensor_bitboards = torch.from_numpy(bitboards).type(torch.FloatTensor)
        return self.autoencoder(tensor_bitboards)

    def apply_siamese_network(self, features):
        tensor_features = torch.from_numpy(features).type(torch.FloatTensor)
        return self.siamese_net(tensor_features).detach().numpy()

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

             # np.zeros(self.n_squares * self.n_pieces * self.n_sides + self.n_extra_bits, dtype=np.int8)
        bitboard = np.zeros(N_SQUARES * N_PIECES * N_SIDES + N_EXTRA_BITS, dtype=np.int8)

        for square in range(N_SQUARES):
            current_piece = board.piece_at(square)

            if current_piece:
                # board.color_at() returns True if White, False if Black
                # Add +1, therefore if White, 1 will be placed at an even position or 1 will be placed at odd position
                current_color = int(board.color_at(square)) + 1
                bitboard[((current_piece.piece_type - 1) + square * 6) * current_color] = 1

        bitboard = self.fill_extra_bits(bitboard, board)
        return bitboard

    def compare_moves(self, move_1, move_2):
        bitboard_1, bitboard_2 = self.get_bitboard(move_1), self.get_bitboard(move_2)
        features_1 = self.extract_features(bitboard_1)[1].detach()
        features_2 =  self.extract_features(bitboard_2)[1].detach()
        group_features = np.hstack((features_1, features_2))
        return self.apply_siamese_network(group_features)[0][0]

    def alphabeta(self, board, depth, alpha, beta, white, orig_board):
        try:
            move = chess.polyglot.MemoryMappedReader('NeuralNetwork/books/Perfect2017-SF12.bin').weighted_choice(board=orig_board).move
            print('book')
            return None, move
        except:
            if depth == 0:
                return self.compare_moves(board, orig_board), None
            if white:
                v = -100 # very (relatively) small number
                moves = board.generate_legal_moves()
                moves = list(moves)
                best_move = None
                for move in moves:
                    new_board = board.copy()
                    new_board.push(move)
                    candidate_v, _ = self.alphabeta(new_board, depth - 1, alpha, beta, False, orig_board)
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
                v = 100 # very (relatively) large number
                moves = board.generate_legal_moves()
                moves = list(moves)
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

    def play_against_stockfish(self):
        stockfish = Stockfish('NeuralNetwork/engines/stockfish-12/stockfish.exe',
                              parameters={"Threads": 4, "Skill Level": 1})

        game = chess.pgn.Game()
        game.headers["Event"] = "Test"
        game.headers["Site"] = "Hammad's PC"
        game.headers["Date"] = str(datetime.now().date())
        game.headers["Round"] = '3'
        game.headers["White"] = "DeepYed"
        game.headers["Black"] = 'Stockfish'

        board = chess.Board()
        history = []
        while not board.is_game_over():
            if board.turn:
                print("DeepYed's Turn")
                # board, depth, alpha, beta, white, orig_board
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


    def get_move_stockfish(self, board, stockfish):
        fen = board.fen()
        stockfish.set_fen_position(fen)

        return stockfish.get_best_move_time(100)


if __name__ == "__main__":
    experiment = Experiment('NeuralNetwork/Checkpoints/april_autoencoder_34.pth.tar',
                            'NeuralNetwork/Checkpoints/april_siamese_7.pth.tar')

    experiment.play_against_stockfish()
