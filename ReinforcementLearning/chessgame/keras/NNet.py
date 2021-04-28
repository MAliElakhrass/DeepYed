import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ReinforcementLearning.chessgame.keras.ChessNNet import ChessNNet as cnnet
from ReinforcementLearning.utils import dotdict
import numpy as np


# CONSTANTS
NUMBER_SQUARES = 8


class NNetWrapper:
    def __init__(self, game):
        self.game = game
        self.args = dotdict({
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 64,
            'num_channels': 256,
        })
        self.nnet = cnnet(self.game, self.args)
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        input_boards, target_pis, target_vs = list(zip(*examples))

        input_boards_reshaped = []
        for board in input_boards:
            input_boards_reshaped.append(board.reshape((8, 8)))
        input_boards_reshaped = np.asarray(input_boards_reshaped)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards_reshaped, y=[target_pis, target_vs], batch_size=self.args.batch_size,
                            epochs=self.args.epochs)

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        # run
        x = board.reshape((1, 8, 8))
        pi, v = self.nnet.model.predict(x)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='training', filename='checkpoint.h5'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='training', filename='checkpoint.h5'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
