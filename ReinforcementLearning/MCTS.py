from ReinforcementLearning.GameState import GameState
from ReinforcementLearning.NeuralNetwork import NeuralNetwork
from torch import Tensor
from torch.autograd import Variable
import chess
import math
import numpy as np
import random
import time


class Node:
    def __init__(self, board: chess.Board, parent, probability):
        self.board: chess.Board = board
        self.parent = parent
        self.children = {}
        self.actions = None  # A CHANGER

        self.N = 0  # number of times action a has been taken form state a
        self.W = 0  # total value of next state
        self.Q = 0  # mean value of next state
        self.P = probability  # prior proba of selecting action a

        self.extended = False
        self.is_leaf = False


class MCTS:
    def __init__(self, iterations, neural_network: NeuralNetwork, device):
        self.c_puct = 1
        self.iterations = iterations
        self.game_state = GameState()
        self.neural_network = neural_network  # TODO
        self.device = device
        self.node_list = {(str(chess.Board()), None): Node(chess.Board(), None, 1.0)}

    def choose_action(self, node: Node):
        """
        This function will choose the action that maximises Q+U
        :param node:
        :return:
        """
        qu = []
        for action in node.actions:
            node_tmp = node.children[action]

            u = self.c_puct * node_tmp.P * math.sqrt(node_tmp.parent.N) / (1 + node_tmp.N)
            qu.append(u + node_tmp.Q)

        idx = np.argmax(qu)

        return node.actions[idx]

    def extension(self, node: Node, probabilities):
        """
        Continue until a leaf node is reached
        :param probabilities:
        :param node:
        :return:
        """
        node.extended = True
        node.actions = list(node.board.legal_moves)

        for i, action in enumerate(node.actions):
            new_board = node.board.copy()
            new_board.push(action)

            idx = self.game_state.get_idx_from_move(action)

            child_node = Node(new_board, node, probabilities[idx])
            child_node.is_leaf = new_board.is_game_over()

            node.children[action] = child_node

            self.node_list[str(new_board), action] = child_node

        return node

    def backup_edges(self, node: Node, value):
        """
        Each edge that was traversed to get to the leaf node is updated as follows
        N = N + 1
        W = W + value
        Q = W / N
        :param node:
        :param value:
        :return:
        """
        curr_node = node
        curr_node.N += 1
        curr_node.W += value
        curr_node.Q = curr_node.W / curr_node.N

        while curr_node:
            curr_node = curr_node.parent
            if curr_node:
                curr_node.N += 1
                curr_node.W += value
                curr_node.Q = curr_node.W / curr_node.N

                if sum([curr_node.children[act].N for act in curr_node.actions]) != curr_node.N:
                    print('warning (MCTS): total visit time N does not match between parent and children')

    def simulate(self):
        for iteration in range(self.iterations):
            print("MCTS-Iter:", iteration)
            curr_board = chess.Board()
            prev_action = None  # WILL IT WORK? ALWAYS NONE
            curr_node = self.node_list[str(curr_board), prev_action]
            time.sleep(0.1)  # WHY?

            for i in range(600):
                state_tmp = Variable(Tensor(self.game_state.get_state(curr_board).reshape(1, 13, 8, 8))).to(self.device)
                p, v = self.neural_network.policy_net(state_tmp)

                if curr_node.extended:
                    curr_action = self.choose_action(curr_node)
                else:
                    curr_node = self.extension(curr_node, p.data.cpu().numpy()[0])

                    # randomly choose an action
                    curr_action = random.choice(curr_node.actions)

                curr_board.push(curr_action)
                curr_node = curr_node.children[curr_action]

                if str(curr_board) != str(curr_node.board):
                    print('warning (MCTS): board information does not match')

                if curr_node.is_leaf:
                    break

            self.backup_edges(curr_node, v.item())

    def play(self, curr_board: chess.Board, prev_action):
        node = self.node_list[str(curr_board), prev_action]
        pi = np.zeros(8*8*8*8*264-64)

        if node.extended and node.N != 0:
            for action in node.actions:
                node_tmp = node.children[action]
                idx = self.game_state.get_idx_from_move(action)
                pi[idx] += node_tmp.N  # self.temperature = 1

            pi = np.divide(pi, pi.sum())
            idx_action = np.argmax(pi)
            if round(sum(pi), 1) != 1.0:
                print('warning (MCTS extended): Pi.sum()!= 1')

            return self.game_state.get_move(idx_action), pi

        else:
            actions = list(curr_board.legal_moves)
            for i, action in enumerate(actions):
                idx = self.game_state.get_idx_from_move(action)
                pi[idx] += 1/len(actions)

            idx_action = np.random.randint(len(actions))
            if round(sum(pi), 1) != 1.0:
                print('warning (MCTS non_extended): Pi.sum()!= 1')
            node = self.extension(node, pi)

            return actions[idx_action], pi
