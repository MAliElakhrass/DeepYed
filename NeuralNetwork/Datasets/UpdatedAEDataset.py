import torch
from torch.utils.data import Dataset
import memory_profiler
import time

import os
import numpy as np


class UpdatedAEDataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if self.mode == "train":
            self.bitboards_directory = "NeuralNetwork/PreprocessedData/Bitboards/Train/"
            self.bitboards_lenghts = self.get_file_names()
        elif self.mode == "valid":
            self.bitboards_directory = "NeuralNetwork/PreprocessedData/Bitboards/Valid/"
            self.bitboards_lenghts = self.get_file_names()
        elif self.mode == "test":
            self.bitboards_directory = "NeuralNetwork/PreprocessedData/Bitboards/Test/"
            self.bitboards_lenghts = self.get_file_names()
        self.loaded_data = np.load(self.bitboards_directory + str(self.bitboards_lenghts[0]) + ".npy")

    def __getitem__(self, index):
        # It expects a tuple, therefore I added a 0 as the second element since the ground truth
        # is not useful for the autoencoder
        groud_truth = 0
        input = torch.from_numpy(self.loaded_data[index]).type(torch.FloatTensor)

        return (input, groud_truth)

    def __len__(self):

        return self.loaded_data.shape[0]

    def get_file_names(self):
        file_names = []
        print(os.getcwd())
        for file in os.listdir(self.bitboards_directory):
            filename, file_extension = os.path.splitext(file)
            file_names.append(int(filename))
        file_names.sort()

        return file_names





if __name__ == "__main__":
    train_dataset = UpdatedAEDataset(
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
