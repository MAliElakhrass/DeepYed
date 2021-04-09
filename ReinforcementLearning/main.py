from ReinforcementLearning.GameState import GameState
from ReinforcementLearning.MCTS import MCTS
from ReinforcementLearning.NeuralNetwork import NeuralNetwork
import chess.engine
import torch


class DeepYed:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.num_epochs = 100
        self.train_iteration = 1000
        self.learning_rate = 0.0001

        self.NN = NeuralNetwork(self.learning_rate, self.device)
        self.game_state = GameState()
        self.mcts = MCTS(100, self.NN, self.device)

        self.engine = chess.engine.SimpleEngine.popen_uci('./engine/stockfish-12/stockfish.exe')
        self.stockfish_flag = True

    def train(self):
        for epoch in range(self.num_epochs):
            self.mcts.simulate()
            board = chess.Board()
            move_count = 0
            move = None
            states_current, probability = [], []

            # Play one game
            while not board.is_game_over():
                node_tmp = self.mcts.node_list[str(board), move]
                if str(node_tmp.board) != str(board):
                    print('warning: node does not match board')
                    break
                states_current.append(self.game_state.get_state(board).reshape(13, 8, 8).copy())

                move, prob = self.mcts.play(board, move)

                if self.stockfish_flag:
                    result = self.engine.play(board, chess.engine.Limit(time=0.05))
                    move = result.move

                if move not in list(board.legal_moves):
                    print('warning: move is not legal')
                    break

                if round(sum(prob), 1) != 1.0:
                    print('warning: Pi.sum()!= 1')
                    break

                probability.append(prob.copy())

                board.push(move)
                move_count += 1

            f = lambda x: 1 if x == '1-0' else (0 if x == '0-1' else 0.5)
            winner = [f(str(board.result())) for i in range(move_count)]

            buffer = (states_current, probability, winner)
            for itr in range(self.train_iteration):
                loss = self.NN.optimize_model(buffer)
                print("Epoch: %d/%d, Iter: %d/%d, loss: %1.12f" % (
                epoch + 1, self.num_epochs, itr + 1, self.train_iteration, loss.item()))


class Validation:
    def __init__(self):
        self.device = torch.device("cuda:0")

        self.learning_rate = 0.001
        self.NN = NeuralNetwork(self.learning_rate, self.device)

        # Load saved model
        self.NN.load_model()
        self.mcts = MCTS(100, self.NN, self.device)

    def valid(self):
        self.mcts.simulate()
        board = chess.Board()
        move = None

        while not board.is_game_over():
            node_tmp = self.mcts.node_list[str(board), move]
            if str(node_tmp.board) != str(board):
                print('warning: node does not match board')
                break

            move, prob = self.mcts.play(board, move)
            if move not in list(board.legal_moves):
                print(move)
                print('warning: move is not legal')
                break
            if round(sum(prob), 1) != 1.0:
                print('warning: Pi.sum()!= 1')
                break
            board.push(move)

        return board


if __name__ == '__main__':
    # training
    model = DeepYed()
    model.train()

    #valid
    valid = Validation()
    output = valid.valid()

    print(output)
    print(output.result())
