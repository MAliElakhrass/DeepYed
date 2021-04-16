from multiprocessing.context import Process
from ReinforcementLearning.MCTS import self_play
import multiprocessing


class DataGenerator:
    def __init__(self, model_path, num_threads):
        self.num_threads = num_threads
        self.model_path = model_path

    def generate_data(self):
        multiprocessing.set_start_method('spawn', force=True)
        processes = []
        for i in range(self.num_threads):
            p1 = Process(target=self_play, args=(self.model_path, 50, i))
            p1.start()
            processes.append(p1)

        for p1 in processes:
            p1.join()
