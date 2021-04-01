import torch
from torch.utils.data import Dataset
import memory_profiler
import time

import os
import numpy as np
import random


class AprilSiameseDataset(Dataset):
    def __init__(self, data, labels, length=4000000):
        self.length = length
        self.loaded_data = data
        self.loaded_labels = labels
        # list_indices_wins = np.where(self.loaded_labels == 1)
        # list_indices_losses = np.where(self.loaded_labels == -1)
        # self.loaded_wins = np.take(self.loaded_data, list_indices_wins[0], axis=0)
        # self.loaded_losses = np.take(self.loaded_data, list_indices_losses[0], axis=0)
        self.loaded_wins = self.loaded_data[self.loaded_labels == 1]
        self.loaded_losses = self.loaded_data[self.loaded_labels == -1]
        print("test")

    def __getitem__(self, index):
        random_win_index = np.random.randint(0, self.loaded_wins.shape[0])
        winning_move = self.loaded_wins[random_win_index]
        random_loss_index = np.random.randint(0, self.loaded_losses.shape[0])
        losing_move = self.loaded_losses[random_loss_index]
        permutation_order = random.randint(0, 1)

        if permutation_order == 0:
            input_moves = np.hstack((winning_move, losing_move))
            input_moves = torch.from_numpy(input_moves).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
        else:
            input_moves = np.hstack((losing_move, winning_move))
            input_moves = torch.from_numpy(input_moves).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)

        return (input_moves, label)

    def __len__(self):
        return self.length


# if __name__ == "__main__":
    # train_dataset = AprilSiameseDataset(
    #     mode="train")()
    # m1 = memory_profiler.memory_usage()
    # t1 = time.time()
    # train_dataset[213434]
    # t2 = time.time()
    # m2 = memory_profiler.memory_usage()
    # time_diff = t2 - t1
    # mem_diff = m2[0] - m1[0]
    # print(f"It took {time_diff} Secs and {mem_diff} Mb to execute the method")
    #
    # print("tests")
