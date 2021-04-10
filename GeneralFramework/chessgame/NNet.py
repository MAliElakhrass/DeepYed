from GeneralFramework.NeuralNet import NeuralNet
from GeneralFramework.chessgame.ChessNNet import ChessNNet as cnnet
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
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=self.args['batch_size'],
                            epochs=self.args['epochs'])

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
        x = np.zeros(NUMBER_SQUARES * NUMBER_SQUARES, dtype=np.int8)

        for square in range(NUMBER_SQUARES * NUMBER_SQUARES):
            piece: chess.Piece = board.piece_at(square)
            if piece:
                color = piece.color
                col = int(square % 8)
                row = int(square / 8)
                x[row * 8 + col] = -piece.piece_type if color == chess.BLACK else piece.piece_type

        return np.reshape(x, (8, 8))

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
