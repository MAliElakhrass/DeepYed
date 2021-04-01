import torch
from torch.utils.data import Dataset
import memory_profiler
import time

import os
import numpy as np


class MultiplesFilesAEDataset(Dataset):
    def __init__(self, mode="train", length=4000000, chunk_size=400000):
        self.mode = mode
        self.length = length
        self.chunk_size = chunk_size
        if self.mode == "train":
            self.bitboards_directory = "NeuralNetwork/PreprocessedData/Bitboards/Train/"

        elif self.mode == "valid":
            self.bitboards_directory = "NeuralNetwork/PreprocessedData/Bitboards/Valid/"

        elif self.mode == "test":
            self.bitboards_directory = "NeuralNetwork/PreprocessedData/Bitboards/Test/"

        self.bitboards_lenghts = self.get_file_names(self.bitboards_directory)

        self.loaded_data = self.merge_bitboards()

    def __getitem__(self, index):
        # It expects a tuple, therefore I added a 0 as the second element since the ground truth
        # is not useful for the autoencoder
        groud_truth = 0
        input = torch.from_numpy(self.loaded_data[index]).type(torch.FloatTensor)

        return (input, groud_truth)

    def __len__(self):

        return self.length

    def get_file_names(self, directory):
        file_names = []
        print(os.getcwd())
        for file in os.listdir(directory):
            filename, file_extension = os.path.splitext(file)
            if filename != "all_bitboards":
                file_names.append(int(filename))
        file_names.sort()

        return file_names

    def merge_bitboards(self):
        full_bitboards = np.memmap(self.bitboards_directory + "all_bitboards" + ".npy",
                                  dtype='int8', mode='w+', shape=(self.length, 773))
        current_count = 0
        for i, file in enumerate(self.bitboards_lenghts):
            # if i != 0:
            current_file = np.memmap(self.bitboards_directory + str(file) + ".npy", dtype='int8', mode='r',
                                     shape=(self.chunk_size, 773))
            full_bitboards[current_count: current_count + self.chunk_size, :] = current_file
            current_count += self.chunk_size
        return full_bitboards



if __name__ == "__main__":
    train_dataset = MultiplesFilesAEDataset(
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
