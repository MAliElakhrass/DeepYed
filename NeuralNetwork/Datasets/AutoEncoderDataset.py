import torch
from torch.utils.data import Dataset
import memory_profiler
import time

import os
import numpy as np


class AutoEncoderDataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        self.bitboards_directory = "NeuralNetwork/PreprocessedData/Bitboards/"
        self.bitboards_lenghts = self.get_file_names()
        self.loaded_data = []
        self.is_loaded = False
        self.max_size = 0
        self.previous_size = 0

    def __getitem__(self, index):
        self.loaded_data, real_index = self.get_data(index)

        # It expects a tuple, therefore I added a 0 as the second element since the ground truth
        # is not useful for the autoencoder
        groud_truth = 0
        input = torch.from_numpy(self.loaded_data[real_index]).type(torch.FloatTensor)

        return input, groud_truth

    def __len__(self):

        return self.bitboards_lenghts[-1]

    def get_file_names(self):
        file_names = []
        print(os.getcwd())
        for file in os.listdir(self.bitboards_directory):
            filename, file_extension = os.path.splitext(file)
            file_names.append(int(filename))
        file_names.sort()

        return file_names

    def get_data(self, index):
        real_index = 0
        if index < self.max_size and index >= self.previous_size:
            return self.loaded_data, index - self.previous_size
        else:
            for n_bitboards in self.bitboards_lenghts:
                if index >= n_bitboards:
                    self.previous_size = n_bitboards
                    continue
                else:
                    if self.previous_size > index:
                        for bit_len in self.bitboards_lenghts:
                            if bit_len < index:
                                self.previous_size = bit_len
                                break
                    # If index is still smaller, than it's the first file
                    if self.previous_size > index:
                        self.previous_size = 0
                    real_index = index - self.previous_size
                    self.max_size = n_bitboards
                    self.loaded_data = np.load(self.bitboards_directory + str(n_bitboards) + ".npy")
                    break
        return self.loaded_data, real_index

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
