from Architecture.AutoEncoder import AutoEncoder

import torch
import os
import numpy as np

BATCH_SIZE = 5000
N_GAMES = 100000


class PrepareSiameseData:
    def __init__(self, mode):
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


    def merge(self):
        # for file in os.listdir(self.bitboards_directory):
        #     filename, file_extension = os.path.splitext(file)
        # print("Extracting features from " + filename + file_extension)
        # y = np.load('mydata.npy', mmap_mode='r')
        encoded_inputs = []
        current_count = 0
        for i, file in enumerate(self.bitboards_files):
            current_data = np.load(self.bitboards_directory + str(file) + ".npy")
            current_data = np.split(current_data, 20)
            # self.loaded_data = self.merge_data_files()
            for game in current_data:
                _, encoder_output = self.autoencoder(torch.from_numpy(game).type(torch.FloatTensor))
                current_count += game.shape[0]
                # encoded_inputs.append(encoder_output.detach().numpy())
                feature_filepath = self.features_directory + str(current_count) + ".npy"
                np.save(feature_filepath, encoder_output.detach().numpy())

    def get_file_names(self):
        file_names = []
        print(os.getcwd())
        for file in os.listdir(self.bitboards_directory):
            filename, file_extension = os.path.splitext(file)
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
    fe_train = FeatureExtractor(mode="train")
    fe_train.extract()

    fe_val = FeatureExtractor(mode="valid")
    fe_val.extract()

    fe_val = FeatureExtractor(mode="test")
    fe_val.extract()

