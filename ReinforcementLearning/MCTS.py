from ReinforcementLearning.Board import Board
from ReinforcementLearning.NeuralNetwork import NeuralNetwork
import copy
import math
import numpy as np


class UCTNode:
    def __init__(self, game, move, parent):
        self.game = game
        self.move = move
        self.parent = parent

        self.is_expanded = False
        self.action_idxes = []
        self.children = {}
        self.child_number_visits = np.zeros([4672], dtype=np.float32)
        self.child_priors = np.zeros([4672], dtype=np.float32)
        self.child_total_value = np.zeros([4672], dtype=np.float32)

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (
                abs(self.child_priors) / (1 + self.child_number_visits))

    def best_child(self):
        if self.action_idxes:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove

    def decode_n_move_pieces(self, board, move):
        i_pos, f_pos, prom = ed.decode_action(board, move)
        for i, f, p in zip(i_pos, f_pos, prom):
            board.player = self.game.player
            board.move_piece(i, f, p)  # move piece to get next board state s
            a, b = i;
            c, d = f
            if board.current_board[c, d] in ["K", "k"] and abs(
                    d - b) == 2:  # if king moves 2 squares, then move rook too for castling
                if a == 7 and d - b > 0:  # castle kingside for white
                    board.player = self.game.player
                    board.move_piece((7, 7), (7, 5), None)
                if a == 7 and d - b < 0:  # castle queenside for white
                    board.player = self.game.player
                    board.move_piece((7, 0), (7, 3), None)
                if a == 0 and d - b > 0:  # castle kingside for black
                    board.player = self.game.player
                    board.move_piece((0, 7), (0, 5), None)
                if a == 0 and d - b < 0:  # castle queenside for black
                    board.player = self.game.player
                    board.move_piece((0, 0), (0, 3), None)
        return board

    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)  # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board, move)
            self.children[move] = UCTNode(
                copy_board, move, parent=self)
        return self.children[move]

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)


def uct_search(game_state, num_reads, net):
    root = UCTNode(game_state)
    for i in range(num_reads): # PK IL FAIT 777
        leaf = root.select_leaf()

    return None, None


def self_play(model_path, num_games, cpu):
    print(f'Loading Model... {cpu}')
    nn = NeuralNetwork()
    nn.model.load_weights(model_path)
    print(f'Model loaded {cpu}')

    states = []
    for idx in range(0, num_games):
        current_board = Board()

        checkmate = False
        while not checkmate and current_board.move_count <= 100:
            states.append(copy.deepcopy(current_board.current_board_bb))
            board_state = copy.deepcopy(current_board.encode_board())

            # UCT_search(current_board,777,chessnet)
            best_move, root = uct_search(current_board, 777, nn)  # 777?? chessnet jsp c quoi?


