from NeuralNetTest.DeepYedDataGenerator import DeepYedDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import numpy as np


MAX_MOVES = 3000000


class DeepYed:
    def __init__(self):
        self.model = None
        self.encoder = load_model('./weights/encoder.h5')

        # Data
        self.split_data()

    def load_data(self):
        whites = np.zeros((MAX_MOVES, 773), dtype=np.int8)
        blacks = np.zeros((MAX_MOVES, 773), dtype=np.int8)

        white_data_path = "./data/white.npy"
        black_data_path = "./data/black.npy"

        whites[0:MAX_MOVES] = np.load(white_data_path)
        blacks[0:MAX_MOVES] = np.load(black_data_path)

        np.random.shuffle(whites)
        np.random.shuffle(blacks)

        return whites, blacks

    def split_data(self):
        whites_bitboard, blacks_bitboard = self.load_data()

        self.whites_train, self.whites_val = train_test_split(whites_bitboard, test_size=0.2, random_state=47)
        self.blacks_train, self.blacks_val = train_test_split(blacks_bitboard, test_size=0.2, random_state=47)

    def build_neural(self):
        input_size = 773

        input_layer1 = Input(shape=(input_size,))
        input_layer2 = Input(shape=(input_size,))

        encoder_left = self.encoder(input_layer1)
        encoder_right = self.encoder(input_layer2)

        combined = concatenate([encoder_left, encoder_right])

        layer1 = Dense(400, activation='relu')(combined)
        layer2 = Dense(200, activation='relu')(layer1)
        layer3 = Dense(100, activation='relu')(layer2)
        output_layer = Dense(2, activation='sigmoid')(layer3)

        self.model = Model(inputs=[input_layer1, input_layer2], outputs=output_layer)
        self.model.summary()

    def fit(self, epochs=1000, batch_size=256):
        train_generator = DeepYedDataGenerator(batch_size, whites=self.whites_train, blacks=self.blacks_train, train=1)
        val_generator = DeepYedDataGenerator(batch_size, whites=self.whites_val, blacks=self.blacks_val, train=0)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        self.model.summary()

        my_callbacks = [
            EarlyStopping(patience=5),
            # ModelCheckpoint(filepath='./model/model.{epoch:02d}-{val_loss:.2f}.h5')
        ]

        history = self.model.fit_generator(train_generator, validation_data=val_generator, epochs=epochs, shuffle=True, callbacks=my_callbacks)

        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def save(self):
        self.model.save('./model/DeepYed.h5')


if __name__ == '__main__':
    dy = DeepYed()
    dy.build_neural()
    dy.fit()
    dy.save()
