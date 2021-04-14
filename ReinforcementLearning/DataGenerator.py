from multiprocessing.context import Process
from ReinforcementLearning.MCTS import yed
from ReinforcementLearning.NeuralNetwork import NeuralNetwork
import multiprocessing


class DataGenerator:
    def __init__(self, model_path, num_threads):
        self.num_threads = num_threads
        self.nn = NeuralNetwork()
        self.nn.model.load_weights(model_path)

    def generate_data(self):
        multiprocessing.set_start_method('spawn', force=True)
        processes = []
        for i in range(self.num_threads):
            p1 = Process(target=yed, args=(i, self.nn.model))
            p1.start()
            processes.append(p1)

        for p1 in processes:
            p1.join()

