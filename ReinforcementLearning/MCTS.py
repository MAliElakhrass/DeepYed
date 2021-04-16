from ReinforcementLearning.Board import Board
from ReinforcementLearning.NeuralNetwork import NeuralNetwork


def self_play(model_path, num_games, cpu):
    print(f'Loading Model... {cpu}')
    nn = NeuralNetwork()
    nn.model.load_weights(model_path)
    print(f'Model loaded {cpu}')

    for idx in range(0, num_games):
        current_board = Board()
        
