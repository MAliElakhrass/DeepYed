import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from NeuralNetKeras.DeepYedDataGenerator import DeepYedDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

# Constants
MAX_MOVES = 3000000


class DeepYed:
    def __init__(self):
        self.model = None
        print("Load Autoencoder ...")
        self.encoder = load_model('NeuralNetKeras/weights/encoder.h5')
        print("Autoencoder loaded!")

        # Data
        self.split_data()

    def load_data(self):
        """
        This function will load the data (white.npy and black.npy)

        :return:
        """
        print("Load data ....")
        whites = np.zeros((MAX_MOVES, 773), dtype=np.int8)
        blacks = np.zeros((MAX_MOVES, 773), dtype=np.int8)

        white_data_path = "NeuralNetKeras/data/white.npy"
        black_data_path = "NeuralNetKeras/data/black.npy"

        whites[0:MAX_MOVES] = np.load(white_data_path)
        blacks[0:MAX_MOVES] = np.load(black_data_path)

        np.random.shuffle(whites)
        np.random.shuffle(blacks)

        print("Data loaded!")

        return whites, blacks

    def split_data(self):
        """
        This function will split the data. 80% for training et 20% for validation

        :return:
        """
        whites_bitboard, blacks_bitboard = self.load_data()

        self.whites_train, self.whites_val = train_test_split(whites_bitboard, test_size=0.2, random_state=47)
        self.blacks_train, self.blacks_val = train_test_split(blacks_bitboard, test_size=0.2, random_state=47)

    def build_neural(self):
        """
        Build the model architecture

        :return:
        """
        input_size = 773

        input_layer1 = Input(shape=(input_size,))
        input_layer2 = Input(shape=(input_size,))

        encoder_left = self.encoder(input_layer1)
        encoder_right = self.encoder(input_layer2)

        combined = concatenate([encoder_left, encoder_right])

        layer1 = Dense(400, activation='relu', name='layer1')(combined)
        layer2 = Dense(200, activation='relu', name='layer2')(layer1)
        layer3 = Dense(100, activation='relu', name='layer3')(layer2)
        output_layer = Dense(2, activation='sigmoid', name='layerOutput')(layer3)

        self.model = Model(inputs=[input_layer1, input_layer2], outputs=output_layer)
        self.model.summary()

    def fit(self, epochs=1000, batch_size=256):
        """
        Fit the model on 1000 epochs or until it overfits

        :param epochs:
        :param batch_size:
        :return:
        """
        train_generator = DeepYedDataGenerator(batch_size, whites=self.whites_train, blacks=self.blacks_train, train=1)
        val_generator = DeepYedDataGenerator(batch_size, whites=self.whites_val, blacks=self.blacks_val, train=0)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        self.model.summary()

        plot_model(self.model, to_file='NeuralNetKeras/figures/model_siamese.png', show_shapes=True,
                   show_layer_names=True)

        my_callbacks = [
            EarlyStopping(patience=5),
            ModelCheckpoint(filepath='NeuralNetKeras/model/model.{epoch:02d}-{val_loss:.2f}.h5')
        ]

        history = self.model.fit_generator(train_generator, validation_data=val_generator, epochs=epochs, shuffle=True,
                                           callbacks=my_callbacks)

        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig('NeuralNetKeras/figures/accuracy_siamese.png')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig('NeuralNetKeras/figures/loss_siamese.png')
        plt.show()

    def save(self):
        """
        Save the model

        :return:
        """
        self.model.save('NeuralNetKeras/model/DeepYed.h5')


if __name__ == '__main__':
    dy = DeepYed()
    dy.build_neural()
    dy.fit()
    dy.save()
