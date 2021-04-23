from tensorflow.keras.utils import Sequence
import numpy as np


class DeepYedDataGenerator(Sequence):
    """
    This class will be used for real-time data feeding to our Keras model
    """
    def __init__(self, batch_size, whites, blacks, train=1):
        self.batch_size = batch_size
        self.train = train
        self.whites = whites
        self.blacks = blacks
        if train:
            self.data_size = 2700000
        else:
            self.data_size = 300000

    def __len__(self):
        """
        Function that returns the number of batches per epoch

        :return:
        """
        return int(np.floor(self.data_size / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data

        :param index:
        :return:
        """
        start_index = index * self.batch_size

        try:
            X_white_batch = self.whites[start_index: start_index + self.batch_size]
            X_black_batch = self.blacks[start_index: start_index + self.batch_size]
        except IndexError:
            X_white_batch = self.whites[start_index:]
            X_black_batch = self.blacks[start_index:]

        # Generate labels
        Y_white_batch = np.ones((X_white_batch.shape[0],))
        Y_black_batch = np.zeros((X_black_batch.shape[0],))

        # Create arrays for batchs
        x = np.stack([X_white_batch, X_black_batch], axis=1)
        labels = np.stack([Y_white_batch, Y_black_batch], axis=1)

        # Randomly switch white and black board
        swap_indices = np.random.randint(2, size=x.shape[0])
        x[swap_indices == 1] = np.flip(x[swap_indices == 1], axis=1)
        labels[swap_indices == 1] = np.flip(labels[swap_indices == 1], axis=1)

        # Split into two numpy arrays to pass into model
        left_batch, right_batch = np.split(x, 2, axis=1)

        left_batch = np.squeeze(left_batch)
        right_batch = np.squeeze(right_batch)

        return [left_batch, right_batch], labels

    def on_epoch_end(self):
        """
        Shuffle the data after each epoch

        :return:
        """
        np.random.shuffle(self.whites)
        np.random.shuffle(self.blacks)
