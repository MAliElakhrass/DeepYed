from ReinforcementLearning.NeuralNetwork import NeuralNetwork
from tensorflow.python.distribute.multi_process_lib import Process
import os


if __name__ == '__main__':
    for i in range(10):
        # Load first model
        model_path = f'model_{i}.pth.rar'

        net = NeuralNetwork()

        current_model = os.path.join('./model_data/', model_path)

        processes_data = []
        for j in range(4):
            p_data = Process()
