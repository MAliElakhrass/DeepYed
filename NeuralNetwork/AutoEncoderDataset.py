import torch
from torch.utils.data import Dataset

import os
import numpy as np


class AutoEncoderDataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        self.bitboards_directory = "./PreprocessedData/Bitboards/"
        self.bitboards_lenghts = self.get_file_names()

    def __getitem__(self, index):
        self.data, real_index = self.get_data(index)

        # It expects a tuple, therefore I added a 0 as the second element since the ground truth
        # is not useful for the autoencoder
        groud_truth = 0
        input = torch.from_numpy(self.data[real_index]).type(torch.FloatTensor)

        return (input, groud_truth)

    def __len__(self):

        return self.bitboards_lenghts[-1]

    def get_file_names(self):
        file_names = []
        for file in os.listdir(self.bitboards_directory):
            filename, file_extension = os.path.splitext(file)
            file_names.append(int(filename))
        file_names.sort()

        return file_names

    def get_data(self, index):
        bitboards = []
        sum_to_index = 0
        real_index = 0
        for n_bitboards in self.bitboards_lenghts:
            if index > n_bitboards:
                sum_to_index = n_bitboards
                continue
            else:
                real_index = index - sum_to_index - 1
                bitboards = np.load(self.bitboards_directory + str(n_bitboards) + ".npy", allow_pickle=True)
                break
        return bitboards, real_index

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


# if __name__ == "__main__":
#     bitboards, real_index = get_data(4026270)
#     print("tests")
