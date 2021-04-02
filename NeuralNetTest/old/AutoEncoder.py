from NeuralNetTest.DenseTied import DenseTied
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model
import gc
import numpy as np


class AutoEncoder:
    def __init__(self):
        self.positions = []
        self.positions_val = []
        self.model = None
        self.encoder = None
        self.decoder = None

    def __encoder(self):
        input_layer = Input(shape=(773,))
        hidden_1 = Dense(600, activation='relu')(input_layer)
        hidden_2 = Dense(400, activation='relu')(hidden_1)
        hidden_3 = Dense(200, activation='relu')(hidden_2)
        code = Dense(100, activation='relu')(hidden_3)

        encoder = Model(input_layer, code, name='encoder')
        encoder.summary()
        self.encoder = encoder
        return encoder

    def __decoder(self):
        code_input = Input(shape=(100,))
        hidden_1 = DenseTied(200, activation='relu', tied_to=self.encoder.layers[4])(code_input)
        hidden_2 = DenseTied(400, activation='relu', tied_to=self.encoder.layers[3])(hidden_1)
        hidden_3 = DenseTied(600, activation='relu', tied_to=self.encoder.layers[2])(hidden_2)
        output_layer = DenseTied(773, activation='sigmoid', tied_to=self.encoder.layers[1])(hidden_3)

        decoder = Model(code_input, output_layer, name='decoder')
        decoder.summary()
        self.decoder = decoder
        return decoder

    def encoder_decoder(self, load=0):
        input_layer = Input(shape=(773,))
        if load:
            self.encoder = load_model('./Pos2Vec/encoder_v1/encoder_epoch66')
        else:
            self.__encoder()
        self.__decoder()

        ec_out = self.encoder(input_layer)
        dc_out = self.decoder(ec_out)

        autoencoder = Model(input_layer, dc_out, name='autoencoder')
        self.model = autoencoder
        self.model.summary()
        return autoencoder

    def train(self, batch_size=256, epochs=20):
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.load_data3()
        for epoch in range(epochs):
            self.shuffle_positions()
            gc.collect()
            train = self.positions[:2000000]
            self.model.fit(train, train, validation_data=(self.positions_val, self.positions_val), epochs=1, batch_size=batch_size)
            train = []
            gc.collect()

            print('Saving ./Pos2Vec/encoder_v1/encoder_epoch' + str(epoch+1))
            self.encoder.save('./Pos2Vec/encoder_v1/encoder_epoch' + str(epoch+1))

    def save(self):
        self.encoder.save('./weights3/encoder_v8.h5')
        # self.decoder.save('./weights3/decoder_v7.h5')
        # self.model.save('./weights3/autoencoder_v8.h5')

    def load_data3(self):
        positions = 2400000-100000
        val_positions = 100000
        num_per_file = 100000
        self.positions = np.zeros((2*positions, 773), dtype='float32')
        self.positions_val = np.zeros((2*val_positions, 773), dtype='float32')
        for i in range(int(positions/num_per_file)):
            print(i + 1)
            start = i*num_per_file
            self.positions[start:start + num_per_file] = np.load('./data5/white_train' + str(i + 1) + '.npy')
            self.positions[positions + start:positions + start + num_per_file] = np.load('./data5/black_train' + str(i + 1) + '.npy')
            if i < val_positions/num_per_file:
                self.positions_val[start:start + num_per_file] = np.load('./data5/white_val' + str(i + 1) + '.npy')
                self.positions_val[val_positions + start:val_positions + start + num_per_file] = np.load('./data5/black_val' + str(i + 1) + '.npy')

    def shuffle_positions(self):
        print("Shuffling positions")
        np.random.shuffle(self.positions)
        gc.collect()

    def predict(self, data):
        return self.encoder.predict(data)


if __name__ == '__main__':
    ae = AutoEncoder()
    ae.encoder_decoder(load=0)
    ae.train(batch_size=256, epochs=100)
    ae.save()
