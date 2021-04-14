from ReinforcementLearning.DataGenerator import DataGenerator
import os


if __name__ == '__main__':
    for i in range(10):
        # Load first model
        model_path = f'model_{i}.h5'
        current_model = os.path.join('./model_data/', model_path)

        dg = DataGenerator(current_model, 8)

        dg.generate_data()

        del dg
