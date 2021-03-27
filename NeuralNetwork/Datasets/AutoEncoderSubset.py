import torch
from torch.utils.data import Dataset
import memory_profiler
import time

import os
import numpy as np
import random


class AutoEncoderSubset(Dataset):
    def __init__(self, mode="train", length=2000000):
        self.length = length
        self.mode = mode
        if self.mode == "train":
            self.bitboards_directory = "NeuralNetwork/PreprocessedData/Bitboards/Train/"
            self.bitboards_lenghts = self.get_file_names()
        elif self.mode == "valid":
            self.bitboards_directory = "NeuralNetwork/PreprocessedData/Bitboards/Valid/"
            self.bitboards_lenghts = self.get_file_names()
        self.loaded_data = []
        self.is_done = False
        self.indices_list = []
        self.load_random_file()

    def __getitem__(self, index):
        current_input = self.get_data()
        if not self.indices_list:
            self.is_done = True

        # It expects a tuple, therefore I added a 0 as the second element since the ground truth
        # is not useful for the autoencoder
        groud_truth = 0
        input = torch.from_numpy(current_input).type(torch.FloatTensor)

        return input, groud_truth

    def __len__(self):

        return self.length

    def get_file_names(self):
        file_names = []
        print(os.getcwd())
        for file in os.listdir(self.bitboards_directory):
            filename, file_extension = os.path.splitext(file)
            file_names.append(int(filename))
        file_names.sort()

        return file_names

    def get_data(self):
        if not self.is_done and self.indices_list:
            index = random.choice(self.indices_list)
            self.indices_list.remove(index)
        else:
            self.load_random_file()
            index = random.choice(self.indices_list)
            self.indices_list.remove(index)
        return self.loaded_data[index]

    def load_random_file(self):
        picked_file = random.choice(self.bitboards_lenghts)
        self.loaded_data = np.load(self.bitboards_directory + str(picked_file) + ".npy")
        self.indices_list = list(range(0, self.loaded_data.shape[0]))
        self.bitboards_lenghts.remove(picked_file)
        self.is_done = False


# bitboards_directory = "./PreprocessedData/Bitboards/"
#
#
# def get_file_names():
#     file_names = []
#     for file in os.listdir(bitboards_directory):
#         filename, file_extension = os.path.splitext(file)
#         file_names.append(int(filename))
#
#     return file_names
#
#
# def get_data(index):
#     file_names = get_file_names()
#     file_names.sort()
#     bitboards = []
#     sum_to_index = 0
#     real_index = 0
#     for n_bitboards in file_names:
#         if index > n_bitboards:
#             sum_to_index = n_bitboards
#             continue
#         else:
#             real_index = index - sum_to_index - 1
#             bitboards = np.load(bitboards_directory + str(n_bitboards) + ".npy", allow_pickle=True)
#             break
#     return bitboards, real_index


if __name__ == "__main__":
    train_dataset = AutoEncoderDataset(
        mode="train")()
    m1 = memory_profiler.memory_usage()
    t1 = time.time()
    data, real_index = train_dataset.get_data(4026270)()
    t2 = time.time()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} Secs and {mem_diff} Mb to execute the method")

    print("tests")
