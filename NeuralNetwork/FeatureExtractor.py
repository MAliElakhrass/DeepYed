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
        self.state = torch.load(
            'NeuralNetwork/Checkpoints/New_regular_AE/lr_5_decay_98/new_reg_autoencoder_199.pth.tar',
            map_location=lambda storage, loc: storage)
        self.autoencoder.load_state_dict(self.state['state_dict'])
        self.mode = mode
        if self.mode == "train":
            self.features_directory = 'NeuralNetwork/PreprocessedData/Features/Train/'
            self.bitboards_directory = 'NeuralNetwork/PreprocessedData/Bitboards/Train/'
        elif self.mode == "valid":
            self.features_directory = 'NeuralNetwork/PreprocessedData/Features/Valid/'
            self.bitboards_directory = 'NeuralNetwork/PreprocessedData/Bitboards/Valid/'
        elif self.mode == "test":
            self.features_directory = 'NeuralNetwork/PreprocessedData/Features/Test/'
            self.bitboards_directory = 'NeuralNetwork/PreprocessedData/Bitboards/Test/'
        self.bach_size = BATCH_SIZE
        self.n_games = N_GAMES
        self.bitboards_files = self.get_file_names()

    def extract(self):
        # for file in os.listdir(self.bitboards_directory):
        #     filename, file_extension = os.path.splitext(file)
        # print("Extracting features from " + filename + file_extension)
        # y = np.load('mydata.npy', mmap_mode='r')
        encoded_inputs = []
        current_count = 0
        for i, file in enumerate(self.bitboards_files):
            current_data = np.load(self.bitboards_directory + str(file) + ".npy")
            current_data = np.split(current_data, 2)
            feat_games = []
            # self.loaded_data = self.merge_data_files()
            for game in current_data:
                _, encoder_output = self.autoencoder(torch.from_numpy(game).type(torch.FloatTensor))
                current_count += game.shape[0]
                feat_games.append(encoder_output.detach().numpy())
            feature_filepath = self.features_directory + str(current_count) + ".npy"
            np.save(feature_filepath, np.vstack(feat_games))

    def get_file_names(self):
        file_names = []
        print(os.getcwd())
        for file in os.listdir(self.bitboards_directory):
            filename, file_extension = os.path.splitext(file)
            if filename != "all_bitboards":
                file_names.append(int(filename))
        file_names.sort()

        return file_names

    def merge_data_files(self):
        for i, file in enumerate(self.bitboards_files):
            if i != 0:
                current_data = np.load(self.bitboards_directory + str(file) + ".npy")
                self.loaded_data = current_data
                # self.loaded_data = np.vstack((self.loaded_data, current_data))
            # else:
            #     self.loaded_data = current_data
        return self.loaded_data


if __name__ == "__main__":
    fe = FeatureExtractor()
    fe.extract()

    print("test")