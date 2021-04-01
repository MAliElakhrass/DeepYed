from Architecture.ImprovedAutoEncoder import ImprovedAutoEncoder
from Architecture.ImprovedSiamese import ImprovedSiamese
from Preprocessing.AprilPreprocessing import AprilPreprocessing

import torch
import numpy as np


class Experiment():
    def __init__(self, DBN_model_path, siamese_model_path):
        self.DBN_model_path = DBN_model_path
        self.siamese_model_path = siamese_model_path
        device = torch.device("cuda")
        self.autoencoder = ImprovedAutoEncoder().to(device)
        self.siamese_net = ImprovedSiamese().to(device)
        self.load_model_states()
        self.autoencoder.eval()
        self.siamese_net.eval()
        self.preprocessor = AprilPreprocessing()

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

    # def extract_bitboards(self, board):
    #     all_moves = list(board.generate_legal_moves())
    #     bitboards = []
    #     for move in all_moves():
    #         copy_board = board.copy()
    #         copy_board.push(move)
    #         current_bitboard = self.preprocessor.get_bitboard(copy_board)
    #         bitboards.append(current_bitboard)


    def compare_moves(self, move_1, move_2):
        bitboard_1, bitboard_2 = self.preprocessor.get_bitboard(move_1), self.preprocessor.get_bitboard(move_2)
        features_1 = self.extract_features(bitboard_1)[1].detach()
        features_2 =  self.extract_features(bitboard_2)[1].detach()
        group_features = np.hstack((features_1, features_2))
        return self.apply_siamese_network(group_features)[0][0]

    def alphabeta(self, board, depth, alpha, beta, white, orig_board):
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

if __name__ == "__main__":
    experiment = Experiment()
    experiment.alphabeta(board, 3, -100, 100, True, board)