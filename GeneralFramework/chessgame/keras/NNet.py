from GeneralFramework.NeuralNet import NeuralNet
from GeneralFramework.chessgame.keras.ChessNNet import ChessNNet as cnnet
import chess
import numpy as np
import os


# CONSTANTS
NUMBER_SQUARES = 8


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        super().__init__(game)
        self.game = game
        self.args = {
            'lr': 0.001,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 64,
            'num_channels': 512,
        }
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

        input_boards = [self.get_bitboard(board) for board in input_boards]

        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_pis = np.delete(target_pis, 4096, axis=1)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=self.args['batch_size'], epochs=self.args['epochs'])

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        # preparing input
        board = self.get_bitboard(board)
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    @staticmethod
    def get_bitboard(board: chess.Board):
        """
            Convert a board to numpy array of size 8x8
            :param board:
            :return: numpy array 8*8
        """
        x = np.zeros(64, dtype=np.int8)
        # print('Flipping: ', flip)
        for pos in range(64):
            piece = board.piece_type_at(pos)  # Gets the piece type at the given square. 0==>blank,1,2,3,4,5,6
            if piece:
                color = int(
                    bool(board.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))  # to check if piece is black or white
                col = int(pos % 8)
                row = int(pos / 8)
                x[row * 8 + col] = -piece if color else piece

        return np.reshape(x, (8, 8))

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
