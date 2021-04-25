import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from DenseTied import DenseTied
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

# CONSTANTS
MAX_MOVES = 3000000


class AutoEncoder:
    def __init__(self):
        self.load_data()

        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    def load_data(self):
        """
        This function will load the data (white.npy and black.npy)

        :return:
        """
        print('Start Loading Data')
        white_data_path = "NeuralNetKeras/data/white.npy"
        black_data_path = "NeuralNetKeras/data/black.npy"

        data = np.zeros((MAX_MOVES * 2, 773), dtype=np.int8)
        data[:MAX_MOVES] = np.load(white_data_path)
        data[MAX_MOVES:] = np.load(black_data_path)

        np.random.shuffle(data)

        self.X_train, self.X_val = train_test_split(data, test_size=0.20, random_state=47)
        print('Loading Data Done!')

    def __encoder(self):
        """
        Encoder model

        :return:
        """
        input_layer = Input(shape=(773,))
        hidden_1 = Dense(600, activation='relu')(input_layer)
        hidden_2 = Dense(400, activation='relu')(hidden_1)
        hidden_3 = Dense(200, activation='relu')(hidden_2)
        output = Dense(100, activation='relu')(hidden_3)

        self.encoder_model = Model(input_layer, output, name='encoder')
        self.encoder_model.summary()

        plot_model(self.encoder_model, to_file='NeuralNetKeras/figures/model_encoder.png', show_shapes=True,
                   show_layer_names=True)

        return self.encoder_model

    def __decoder(self):
        """
        Decoder model

        :return:
        """
        input_layer = Input(shape=(100,))
        hidden_1 = DenseTied(200, activation='relu', tied_to=self.encoder_model.layers[4])(input_layer)
        hidden_2 = DenseTied(400, activation='relu', tied_to=self.encoder_model.layers[3])(hidden_1)
        hidden_3 = DenseTied(600, activation='relu', tied_to=self.encoder_model.layers[2])(hidden_2)
        output = DenseTied(773, activation='sigmoid', tied_to=self.encoder_model.layers[1])(hidden_3)

        self.decoder_model = Model(input_layer, output, name='decoder')
        self.decoder_model.summary()

        plot_model(self.decoder_model, to_file='NeuralNetKeras/figures/model_decoder.png', show_shapes=True,
                   show_layer_names=True)

        return self.decoder_model

    def encoder_decoder(self, load=False, checkpoint=None):
        """
        Autoencoder model

        :param checkpoint: Checkpoint to load model from
        :param load: If True, load the encoder model and continue training
        :return:
        """
        input_layer = Input(shape=(773,))
        if load:
            self.encoder_model = load_model(f'NeuralNetKeras/weights/model.{checkpoint}')
        else:
            self.__encoder()
        self.__decoder()

        ec_out = self.encoder_model(input_layer)
        dc_out = self.decoder_model(ec_out)

        self.model = Model(input_layer, dc_out, name='autoencoder')
        self.model.summary()

        plot_model(self.model, to_file='NeuralNetKeras/figures/model_ae.png', show_shapes=True, show_layer_names=True)

    def train(self, batch_size=256, epochs=200):
        """
        Train the model until the epochs are reached or the early stopping is triggered.

        :param batch_size: Size of the batches
        :param epochs: Number of epochs to train
        :return:
        """
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

        my_callbacks = [
            EarlyStopping(patience=3),
            ModelCheckpoint(filepath='NeuralNetKeras/weights/model.{epoch:02d}-{val_loss:.2f}', save_format='tf')
        ]

        history = self.model.fit(self.X_train, self.X_train, validation_data=(self.X_val, self.X_val), epochs=epochs,
                                 batch_size=batch_size, callbacks=my_callbacks, verbose=1)

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig('NeuralNetKeras/figures/accuracy_ae.png')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig('NeuralNetKeras/figures/loss_ae.png')
        plt.show()

    def save(self):
        """
        Save the encoder model

        :return:
        """
        self.encoder_model.save('NeuralNetKeras/weights/encoder.h5')


if __name__ == '__main__':
    ae = AutoEncoder()
    ae.encoder_decoder(load=False)
    ae.train(batch_size=256, epochs=200)
    ae.save()

