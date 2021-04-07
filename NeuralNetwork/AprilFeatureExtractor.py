# from Architecture.ImprovedAutoEncoder import ImprovedAutoEncoder
from Architecture.AutoEncoder import AutoEncoder

import torch
import os
import numpy as np
import re
import pandas as pd

BATCH_SIZE = 40


class AprilFeatureExtractor:
    def __init__(self):
        self.autoencoder = AutoEncoder()
        self.results_files = self.get_file_names()
        self.val_losses_file = self.results_files[-2]
        self.val_file = pd.read_csv("NeuralNetwork/Results/Jdid_Reg_AE/" + self.val_losses_file + ".csv", header=None)
        self.best_epoch = self.val_file.idxmin().values[0]
        # self.best_epoch = 17
        # Load the best epoch
        self.state = torch.load(
            'NeuralNetwork/Checkpoints/Jdid_Reg_AE/lr_5_decay_98/april_autoencoder_' + str(
                self.best_epoch) + '.pth.tar',
            map_location=lambda storage, loc: storage)
        self.batch_size = BATCH_SIZE

        self.loaded_bitboards = np.load("NeuralNetwork\\PreprocessedData\\Bitboards\\all_bitboards.npy")
        self.chunked_bitboards = np.split(self.loaded_bitboards, self.batch_size)

    def tryint(self, s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(self, s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [self.tryint(c) for c in re.split('([0-9]+)', s)]

    def get_file_names(self):
        file_names = []
        print(os.getcwd())
        for file in os.listdir("NeuralNetwork/Results/Jdid_Reg_AE"):
            filename, file_extension = os.path.splitext(file)
            file_names.append(filename)
        file_names.sort(key=self.alphanum_key)
        return file_names

    def extract(self):
        encoded_inputs = []
        for bitboard in self.chunked_bitboards:
            tensor_bitboard = torch.from_numpy(bitboard).type(torch.FloatTensor)
            _, encoder_output = self.autoencoder(tensor_bitboard)
            encoded_inputs.append(encoder_output.detach().numpy())
        full_features = np.vstack(encoded_inputs)
        np.save("NeuralNetwork\\PreprocessedData\\Features\\reg_features.npy", full_features)


if __name__ == "__main__":
    fe_extractor = AprilFeatureExtractor()
    fe_extractor.extract()
