import os
import numpy as np


class WinLossCounter:
    def __init__(self, mode):
        self.mode = mode
        self.loss_count = 0
        self.win_count = 0
        if self.mode == "train":
            self.features_directory = "NeuralNetwork/PreprocessedData/Bitboards/Train/"
            self.labels_directory = "NeuralNetwork/PreprocessedData/Labels/Train/"
        elif self.mode == "valid":
            self.features_directory = "NeuralNetwork/PreprocessedData/Bitboards/Valid/"
            self.labels_directory = "NeuralNetwork/PreprocessedData/Labels/Valid/"
        elif self.mode == "test":
            self.features_directory = "NeuralNetwork/PreprocessedData/Bitboards/Test/"
            self.labels_directory = "NeuralNetwork/PreprocessedData/Labels/Test/"
        self.files_list = self.get_file_names()

    def get_file_names(self):
        file_names = []
        print(os.getcwd())
        for file in os.listdir(self.features_directory):
            filename, file_extension = os.path.splitext(file)
            file_names.append(int(filename))
        file_names.sort()

        return file_names


    def get_W_L_counts(self):
        for file in self.files_list:
            # loaded_data = np.load(
            #     self.features_directory + str(file) + ".npy")
            loaded_labels = np.load(
                self.labels_directory + str(file) + ".npy")
            # loaded_wins = loaded_data[loaded_labels == 1]
            # loaded_losses = loaded_data[loaded_labels == -1]
            self.loss_count += np.count_nonzero(loaded_labels == -1)
            self.win_count += np.count_nonzero(loaded_labels == 1)
        return self.loss_count, self.win_count

    def get_W_L_first_file(self):
        loaded_labels = np.load(
            self.labels_directory + str(self.files_list[0]) + ".npy")
        self.loss_count += np.count_nonzero(loaded_labels == -1)
        self.win_count += np.count_nonzero(loaded_labels == 1)
        return self.loss_count, self.win_count

if __name__ == "__main__":
    train_W_L_counter = WinLossCounter(mode="train")
    c_losses, c_wins = train_W_L_counter.get_W_L_counts()
    # c_losses, c_wins = train_W_L_counter.get_W_L_first_file()
    print("Number of losses for training set : " + str(c_losses))
    print("Number of wins for training set : " + str(c_wins))
    valid_W_L_counter = WinLossCounter(mode="valid")
    # c_losses, c_wins = valid_W_L_counter.get_W_L_first_file()
    c_losses, c_wins = valid_W_L_counter.get_W_L_counts()
    print("Number of losses for validation set : " + str(c_losses))
    print("Number of wins for validation set : " + str(c_wins))

    test_W_L_counter = WinLossCounter(mode="test")
    # c_losses, c_wins = test_W_L_counter.get_W_L_first_file()
    c_losses, c_wins = test_W_L_counter.get_W_L_counts()
    print("Number of losses for test set : " + str(c_losses))
    print("Number of wins for test set : " + str(c_wins))

