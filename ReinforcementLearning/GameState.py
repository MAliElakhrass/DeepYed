import chess
import numpy as np


class GameState:
    def __init__(self):
        self.pieces = ['.', 'K', 'Q', 'B', 'N', 'R', 'P', 'k', 'q', 'b', 'n', 'r', 'p']

        self.possibe_actions = self.get_all_possible_state()

    @staticmethod
    def get_all_possible_state():
        """
        This function will get all the possible actions
        :return:
        """
        square_x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        square_y = ['1', '2', '3', '4', '5', '6', '7', '8']
        promotion = ['', 'q']

        return [i + j + m + n + k for k in promotion for i in square_x for j in square_y for m in square_x for n in
                square_y if i + j != m + n]

    def get_state(self, board: chess.Board):
        """
        This function will get the state of a board
        :param board:
        :return:
        """

        state_tmp = []
        for item in board.__str__():
            if item != ' ' and item != '\n':
                state_tmp.append(self.pieces.index(item))

        state_tmp = np.array(state_tmp).reshape(8, 8)

        state = np.zeros((13, 8, 8))
        for i in range(8):
            for j in range(8):
                state[state_tmp[i, j], i, j] = 1

        return state

    def get_move(self, idx):
        """
        Parse a UCI move from the possibe_actions list
        :param idx:
        :return:
        """
        return chess.Move.from_uci(self.possibe_actions[idx])

    def get_idx_from_move(self, move):
        """
        This function will take the index corresponding to the move in the possibe_actions list
        :param move:
        :return:
        """
        if len(str(move)) > 4:
            move = ''.join([char for char in str(move)][0:4]) + 'q'
        else:
            move = str(move)

        return self.possibe_actions.index(move)


if __name__ == '__main__':
    yed = GameState()
    print(len(yed.possibe_actions))
    print(yed.get_move(0))
