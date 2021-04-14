#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class NeuralNetwork:
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.board_x = 8
        self.board_y = 8

        self.num_channels = 256
        self.kernel_size = 3
        self.padding = 'same'
        self.use_bias = False
        self.dropout = 0.3
        self.lr = 0.001

        self.action_size = 8 * 8 * 73

        self.model = self.construct_model()

    def conv_layer(self, x):
        return Activation('relu')(
            BatchNormalization()
            (Conv2D(self.num_channels, self.kernel_size, padding=self.padding, use_bias=self.use_bias)(x)))

    def construct_model(self):
        input_board = Input(name='input', shape=(self.board_x, self.board_y))
        x = Reshape((self.board_x, self.board_y, 1))(input_board)

        for i in range(20):
            x = self.conv_layer(x)

        x_flat = Flatten()(x)

        # OUTPUT LAYER JSP
        s_fc1 = Dropout(self.dropout)(
            Activation('relu')(BatchNormalization()(Dense(1024, use_bias=False)(x_flat))))
        s_fc2 = Dropout(self.dropout)(
            Activation('relu')(BatchNormalization()(Dense(512, use_bias=False)(s_fc1))))

        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        v = Dense(1, activation='tanh', name='v')(s_fc2)

        model = Model(inputs=input_board, outputs=[pi, v])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.lr))

        return model


def init_random():
    nn = NeuralNetwork()
    nn.model.save_weights('./model_data/model_0.h5')


if __name__ == '__main__':
    init_random()
