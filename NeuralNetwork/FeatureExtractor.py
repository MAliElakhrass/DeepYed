from Architecture.AutoEncoder import AutoEncoder

import torch
import os
import numpy as np

BATCH_SIZE = 5000
N_GAMES = 100000


class FeatureExtractor:
    def __init__(self):
        self.autoencoder = AutoEncoder()
        # Load the best epoch
        self.state = torch.load('NeuralNetwork/Checkpoints/AutoEncoder/lr_0_decay_0/autoencoder_1.pth.tar',
                                map_location=lambda storage, loc: storage)
        self.features_directory = 'NeuralNetwork/PreprocessedData/Features/'
        self.bitboards_directory = 'NeuralNetwork/PreprocessedData/Bitboards/'
        self.bach_size = BATCH_SIZE
        self.n_games = N_GAMES

    def extract(self):
        for file in os.listdir(self.bitboards_directory):
            filename, file_extension = os.path.splitext(file)
            self.loaded_bitboards = np.load(self.bitboards_directory + filename + ".npy")
            _, encoder_output = self.autoencoder(torch.from_numpy(self.loaded_bitboards).type(torch.FloatTensor))
            detached_grad_encoder = encoder_output.detach().numpy()
            feature_filepath = self.features_directory + filename + file_extension
            np.save(feature_filepath, np.vstack(detached_grad_encoder))


if __name__ == "__main__":
    fe = FeatureExtractor()
    fe.extract()
