import torch
from torch.utils.data import Dataset
import memory_profiler
import time

import os
import numpy as np


class AprilAEDataset(Dataset):
    def __init__(self, data):
        self.loaded_data = data

    def __getitem__(self, index):
        # It expects a tuple, therefore I added a 0 as the second element since the ground truth
        # is not useful for the autoencoder
        groud_truth = 0
        input = torch.from_numpy(self.loaded_data[index]).type(torch.FloatTensor)

        return (input, groud_truth)

    def __len__(self):
        return self.loaded_data.shape[0]




#
# if __name__ == "__main__":
    # data = np.load()
    # train_dataset = AprilAEDataset(
    #     data=data)()
    # m1 = memory_profiler.memory_usage()
    # t1 = time.time()
    # data, real_index = train_dataset.get_data(4026270)()
    # t2 = time.time()
    # m2 = memory_profiler.memory_usage()
    # time_diff = t2 - t1
    # mem_diff = m2[0] - m1[0]
    # print(f"It took {time_diff} Secs and {mem_diff} Mb to execute the method")
    #
    # print("tests")
