import torch
from torch.utils.data import Dataset
import memory_profiler
import time

import os
import numpy as np
import random


class UpdatedSiameseDataset(Dataset):
    def __init__(self, mode="train", length=4000000, chunk_size=100000):
        self.length = length
        self.chunk_size = chunk_size
        self.mode = mode
        if self.mode == "train":
            self.features_directory = "NeuralNetwork/PreprocessedData/Features/Train/"
            self.labels_directory = "NeuralNetwork/PreprocessedData/Labels/Train/"

        elif self.mode == "valid":
            self.features_directory = "NeuralNetwork/PreprocessedData/Features/Valid/"
            self.labels_directory = "NeuralNetwork/PreprocessedData/Labels/Valid/"

        elif self.mode == "test":
            self.features_directory = "NeuralNetwork/PreprocessedData/Features/Test/"
            self.labels_directory = "NeuralNetwork/PreprocessedData/Labels/Test/"

        self.features_files_length = self.get_file_names(self.features_directory)
        self.labels_files = self.get_file_names(self.labels_directory)

        # self.loaded_data = np.load(self.bitboards_directory + str(self.features_files_length[0]) + ".npy")
        self.loaded_data = self.merge_features()
        # np.load(self.labels + str(self.features_files_length[0]) + ".npy")
        self.loaded_labels = self.merge_labels()
        p = np.random.permutation(len(self.loaded_labels))
        self.loaded_data = self.loaded_data[p]
        self.loaded_labels = self.loaded_labels[p]
        list_indices_wins = np.where(self.loaded_labels == 1)
        list_indices_losses = np.where(self.loaded_labels == -1)
        self.loaded_wins = np.take(self.loaded_data, list_indices_wins[0], axis=0)
        self.loaded_losses = np.take(self.loaded_data, list_indices_losses[0], axis=0)
        print("test")

    def __getitem__(self, index):
        winning_move = self.loaded_wins[
            np.random.randint(0, self.loaded_wins.shape[0])]
        losing_move = self.loaded_losses[
            np.random.randint(0, self.loaded_losses.shape[0])]
        permutation_order = random.randint(0, 1)

        if permutation_order == 0:
            input_moves = np.hstack((losing_move, winning_move))
            input_moves = torch.from_numpy(input_moves).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
            return (input_moves, label)
        else:
            input_moves = np.hstack((winning_move, losing_move))
            input_moves = torch.from_numpy(input_moves).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)
            return (input_moves, label)

    def __len__(self):
        return self.length

    def get_file_names(self, directory):
        file_names = []
        print(os.getcwd())
        for file in os.listdir(directory):
            filename, file_extension = os.path.splitext(file)
            if filename != "all_labels" and filename != "all_features":
                file_names.append(int(filename))
        file_names.sort()

        return file_names

    def merge_features(self):
        full_features = np.memmap(self.features_directory + "all_features" + ".npy",
                                  dtype='float32', mode='w+', shape=(self.length, 100))
        current_count = 0
        for i, file in enumerate(self.features_files_length):
            # if i != 0:
            current_file = np.memmap(self.features_directory + str(file) + ".npy", dtype='float32', mode='r',
                                     shape=(self.chunk_size, 100))
            full_features[current_count: current_count + self.chunk_size, :] = current_file
            current_count += self.chunk_size
        return full_features

    def merge_labels(self):
        full_labels = np.memmap(self.labels_directory + "all_labels" + ".npy", dtype='int8', mode='w+',
                                shape=(self.length, 1))
        current_count = 0
        for i, file in enumerate(self.labels_files):
            # if i != 0:
            current_file = np.memmap(self.labels_directory + str(file) + ".npy", dtype='int8', mode='r',
                                     shape=(self.chunk_size, 1))
            full_labels[current_count:current_count + self.chunk_size, :] = current_file
            current_count += self.chunk_size
        return full_labels


if __name__ == "__main__":
    train_dataset = UpdatedSiameseDataset(
        mode="train")()
    m1 = memory_profiler.memory_usage()
    t1 = time.time()
    train_dataset[213434]
    t2 = time.time()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} Secs and {mem_diff} Mb to execute the method")

    print("tests")
