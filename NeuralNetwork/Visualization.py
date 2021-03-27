import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_val_train(dir_path, save_name):
    epochs = np.array(range(1, 201))
    data_train = pd.read_csv(dir_path + "199_Train_Res_per_epoch.csv", header=None, sep='\\s+')
    data_valid = pd.read_csv("NeuralNetwork/Results/ImprovedAutoEncoder/199_Valid_Res_per_epoch.csv", header=None,
                             sep='\\s+')

    plt.plot(epochs, data_train, color='lightskyblue', label="Train Losses")
    plt.plot(epochs, data_valid, color='darksalmon', label="Validation Losses")
    plt.title("Training and validation loss against the number of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend(loc='center right')
    plt.savefig("NeuralNetwork/Plots/" + save_name + ".png")
    plt.show()


def plot_differences(dir_path, save_name):
    epochs = np.array(range(1, 201))
    dat_differences = pd.read_csv(dir_path + "199_Differences.csv", header=None, sep='\\s+')

    plt.plot(epochs, dat_differences, color='lightskyblue', label="Differences")
    plt.title("Differences evolution against the number of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Differences")
    plt.legend(loc='center right')
    plt.savefig("NeuralNetwork/Plots/" + save_name + ".png")
    plt.show()


if __name__ == "__main__":
    dir_path = "NeuralNetwork/Results/ImprovedAutoEncoder/"
    save_name = "improved_autoencoder_losses"
    plot_val_train(dir_path, save_name)
    plot_differences(dir_path, "improved_autoencoder_differences")

    dir_path = "NeuralNetwork/Results/AutoEncoder/"
    save_name = "autoencoder_losses"
    plot_val_train(dir_path, save_name)
    plot_differences(dir_path, "autoencoder_differences")
