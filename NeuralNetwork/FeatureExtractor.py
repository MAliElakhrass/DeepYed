from Architecture.AutoEncoder import AutoEncoder

import torch
import os
import numpy as np

BATCH_SIZE = 5000
N_GAMES = 100000


class FeatureExtractor:
    def __init__(self, mode):
        self.autoencoder = AutoEncoder()
        # Load the best epoch
        self.state = torch.load('NeuralNetwork/Checkpoints/AutoEncoder/lr_0_decay_0/autoencoder_1.pth.tar',
                                map_location=lambda storage, loc: storage)

        self.mode = mode
        if self.mode == "train":
            self.features_directory = 'NeuralNetwork/PreprocessedData/Features/Train/'
            self.bitboards_directory = 'NeuralNetwork/PreprocessedData/Bitboards/Train/'
        elif self.mode == "valid":
            self.features_directory = 'NeuralNetwork/PreprocessedData/Features/Valid/'
            self.bitboards_directory = 'NeuralNetwork/PreprocessedData/Bitboards/Valid/'
        self.bach_size = BATCH_SIZE
        self.n_games = N_GAMES

    def extract(self):
        for file in os.listdir(self.bitboards_directory):

            filename, file_extension = os.path.splitext(file)
            print("Extracting features from " + filename + file_extension)
            self.loaded_bitboards = np.load(self.bitboards_directory + filename + ".npy")
            _, encoder_output = self.autoencoder(torch.from_numpy(self.loaded_bitboards).type(torch.FloatTensor))
            detached_grad_encoder = encoder_output.detach().numpy()
            feature_filepath = self.features_directory + filename + file_extension
            np.save(feature_filepath, np.vstack(detached_grad_encoder))


if __name__ == "__main__":
    fe_train = FeatureExtractor(mode="train")
    fe_train.extract()

    fe_val = FeatureExtractor(mode="valid")
    fe_val.extract()
