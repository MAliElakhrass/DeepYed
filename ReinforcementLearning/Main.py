from ReinforcementLearning.NeuralNetwork import NeuralNetwork
import os


if __name__ == '__main__':
    for i in range(10):
        # Load first model
        model_path = f'model_{i}.pth.rar'

        net = NeuralNetwork()

        current_model = os.path.join('./model_data/', model_path)
