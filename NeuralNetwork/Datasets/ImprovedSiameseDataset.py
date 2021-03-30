import torch
from torch.utils.data import Dataset
import memory_profiler
import time

import os
import numpy as np
import random


class ImprovedSiameseDataset(Dataset):
    def __init__(self, mode="train", n_inputs=1000000):
        self.mode = mode
        self.n_inputs = n_inputs
        if self.mode == "train":
            self.features_directory = "NeuralNetwork/PreprocessedData/Features/Train/"
            self.labels_directory = "NeuralNetwork/PreprocessedData/Labels/Train/"
        else:
            self.features_directory = "NeuralNetwork/PreprocessedData/Features/Valid/"
            self.labels_directory = "NeuralNetwork/PreprocessedData/Labels/Valid/"
        self.features_files_length = self.get_file_names()
        self.current_filename = self.features_files_length[0]
        self.current_file_idx = 0
        self.previous_file_idx = 0
        self.loaded_data = []
        self.loaded_wins = []
        self.loaded_losses = []
        self.cur_wins, self.cur_losses = 0, 0
        self.load_initial_files()
        self.is_loss_reloaded = False
        self.is_win_reloaded = False
        self.prev_wins_count, self.prev_losses_count = 0, 0


    def __getitem__(self, index):
        permutation_order = random.randint(0, 1)
        self.get_data(index)
        win_index = index - self.prev_wins_count
        loss_index = index - self.prev_losses_count
        winning_move = self.loaded_wins[np.random.randint(0, self.loaded_wins.shape[0])]
        losing_move = self.loaded_losses[np.random.randint(0, self.loaded_losses.shape[0])]
        # Reset all parameters for next epoch
        if index == self.n_inputs - 1:
            self.current_filename = self.features_files_length[0]
            self.current_file_idx = 0
            self.previous_file_idx = 0
            self.loaded_data = []
            self.loaded_wins = []
            self.loaded_losses = []
            self.cur_wins, self.cur_losses = 0, 0
            self.load_initial_files()
            self.is_loss_reloaded = False
            self.is_win_reloaded = False
            self.prev_wins_count, self.prev_losses_count = 0, 0

        if permutation_order == 0:
            input_moves = np.hstack((winning_move, losing_move))
            input_moves = torch.from_numpy(input_moves).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
            return (input_moves, label)
        else:
            input_moves = np.hstack((winning_move, losing_move))
            input_moves = torch.from_numpy(input_moves).type(torch.FloatTensor)
            label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)
            return (input_moves, label)

    def __len__(self):

        return self.n_inputs

    def get_file_names(self):
        file_names = []
        print(os.getcwd())
        for file in os.listdir(self.features_directory):
            filename, file_extension = os.path.splitext(file)
            file_names.append(int(filename))
        file_names.sort()

        return file_names

    def reload_losses(self):
        # if self.current_file_idx + 1 >= len(self.features_files_length):
        #     print("No more losses, all were loaded")
        #     return
        self.loaded_data = np.load(
            self.features_directory + str(self.features_files_length[self.current_file_idx + 1]) + ".npy")
        self.loaded_labels = np.load(
            self.labels_directory + str(self.features_files_length[self.current_file_idx + 1]) + ".npy")
        self.prev_losses_count = self.cur_losses
        self.loaded_losses = self.loaded_data[self.loaded_labels == -1]
        self.is_loss_reloaded = True
        self.cur_losses += self.loaded_losses.shape[0]
        self.set_params()

    def reload_wins(self):
        # if self.current_file_idx + 1 >= len(self.features_files_length):
        #     print("No more wins, all were loaded")
        #     return
        self.loaded_data = np.load(
            self.features_directory + str(self.features_files_length[self.current_file_idx + 1]) + ".npy")
        self.loaded_labels = np.load(
            self.labels_directory + str(self.features_files_length[self.current_file_idx + 1]) + ".npy")
        self.prev_wins_count = self.cur_wins
        self.loaded_wins = self.loaded_data[self.loaded_labels == 1]
        self.cur_wins += self.loaded_wins.shape[0]
        self.is_win_reloaded = True
        self.set_params()

    def set_params(self):
        if self.is_loss_reloaded and self.is_win_reloaded:
            self.current_file_idx += 1
            self.current_filename = self.features_files_length[self.current_file_idx]
            self.is_win_reloaded = False
            self.is_loss_reloaded = False

    def reload_all_files(self):
        # if self.current_file_idx + 1 >= len(self.features_files_length):
        #     print("No more wins and losses, all were loaded")
        #     return
        self.current_file_idx += 1
        self.current_filename = self.features_files_length[self.current_file_idx]
        self.loaded_data = np.load(
            self.features_directory + str(self.current_filename) + ".npy")
        self.loaded_labels = np.load(
            self.labels_directory + str(self.current_filename) + ".npy")

        self.prev_losses_count = self.cur_losses
        self.prev_wins_count = self.cur_wins
        self.cur_losses += self.loaded_losses.shape[0]
        self.cur_wins += self.loaded_wins.shape[0]

    def load_initial_files(self):
        self.current_filename = self.features_files_length[self.current_file_idx]
        self.loaded_data = np.load(
            self.features_directory + str(self.current_filename) + ".npy")
        self.loaded_labels = np.load(
            self.labels_directory + str(self.current_filename) + ".npy")
        self.loaded_wins = self.loaded_data[self.loaded_labels == 1]
        self.cur_wins += self.loaded_wins.shape[0]
        self.loaded_losses = self.loaded_data[self.loaded_labels == -1]
        self.cur_losses += self.loaded_losses.shape[0]

    def get_data(self, index):
        if index >= self.cur_wins and index >= self.cur_losses:
            self.reload_all_files()
        elif index >= self.cur_wins:
            self.reload_wins()
        elif index >= self.cur_losses:
            self.reload_losses()


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
    train_dataset = SiameseDataset(
        mode="train")()
    m1 = memory_profiler.memory_usage()
    t1 = time.time()
    train_dataset[116073]
    t2 = time.time()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} Secs and {mem_diff} Mb to execute the method")

    print("tests")
