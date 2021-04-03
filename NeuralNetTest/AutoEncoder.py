from NeuralNetTest.DenseTied import DenseTied
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
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
        print('Start Loading Data')
        white_data_path = "./data/white.npy"
        black_data_path = "./data/black.npy"

        data = np.zeros((MAX_MOVES * 2, 773), dtype=np.int8)
        data[:MAX_MOVES] = np.load(white_data_path)
        data[MAX_MOVES:] = np.load(black_data_path)

        np.random.shuffle(data)

        self.X_train, self.X_val = train_test_split(data, test_size=0.20, random_state=47)
        print('Loading Data Done!')

    def __encoder(self):
        input_layer = Input(shape=(773,))
        hidden_1 = Dense(600, activation='relu')(input_layer)
        hidden_2 = Dense(400, activation='relu')(hidden_1)
        hidden_3 = Dense(200, activation='relu')(hidden_2)
        output = Dense(100, activation='relu')(hidden_3)

        self.encoder_model = Model(input_layer, output, name='encoder')
        self.encoder_model.summary()

        return self.encoder_model

    def __decoder(self):
        input_layer = Input(shape=(100,))
        hidden_1 = DenseTied(200, activation='relu', tied_to=self.encoder_model.layers[4])(input_layer)
        hidden_2 = DenseTied(400, activation='relu', tied_to=self.encoder_model.layers[3])(hidden_1)
        hidden_3 = DenseTied(600, activation='relu', tied_to=self.encoder_model.layers[2])(hidden_2)
        output = DenseTied(773, activation='sigmoid', tied_to=self.encoder_model.layers[1])(hidden_3)

        self.decoder_model = Model(input_layer, output, name='decoder')
        self.decoder_model.summary()

        return self.decoder_model

    def encoder_decoder(self, load=0):
        input_layer = Input(shape=(773,))
        if load:
            self.encoder_model = load_model('./Pos2Vec/encoder_v1/encoder_epoch66')
        else:
            self.__encoder()
        self.__decoder()

        ec_out = self.encoder_model(input_layer)
        dc_out = self.decoder_model(ec_out)

        self.model = Model(input_layer, dc_out, name='autoencoder')
        self.model.summary()

    def train(self, batch_size=256, epochs=200):
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

        my_callbacks = [
            EarlyStopping(patience=3),
            ModelCheckpoint(filepath='./weights/model.{epoch:02d}-{val_loss:.2f}', save_format='tf')
        ]

        self.model.fit(self.X_train, self.X_train, validation_data=(self.X_val, self.X_val), epochs=epochs,
                       batch_size=batch_size, callbacks=my_callbacks, verbose=1)

    def save(self):
        self.encoder_model.save('./weights/encoder.h5')

    def predict(self, data):
        return self.encoder_model.predict(data)


if __name__ == '__main__':
    ae = AutoEncoder()
    ae.encoder_decoder(load=0)
    ae.train(batch_size=256, epochs=200)
    ae.save()
