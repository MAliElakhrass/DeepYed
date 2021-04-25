import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils import dotdict
from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class ChessNNet:
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.model = self.construct_graph()

    def construct_graph(self):
        input_boards = Input(name='input', shape=(self.board_x, self.board_y))
        x = Reshape([self.board_x, self.board_y, 1])(input_boards)

        # Convolutional Block
        x = Activation('relu')(
            BatchNormalization()(Conv2D(self.args.num_channels, kernel_size=3, strides=1, padding='same')(x)))

        # Res Block
        for i in range(19):
            res = x
            x = Activation('relu')(
                BatchNormalization()(Conv2D(self.args.num_channels, kernel_size=3, strides=1, padding='same',
                                            use_bias=False)(x)))
            x = BatchNormalization()(Conv2D(self.args.num_channels, kernel_size=3, strides=1, padding='same',
                                            use_bias=False)(x))
            x = Add()([x, res])
            x = Activation('relu')(x)

        # Out Block
        # Value Head
        v = Activation('relu')(
            BatchNormalization()(Conv2D(1, kernel_size=1, strides=1, padding='valid')(x)))
        v = Flatten()(v)
        v = Dense(64, activation='relu')(v)
        v = Dense(1, activation='tanh')(v)

        # Policy Head
        pi = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=1, strides=1, padding='valid')(x)))
        pi = Flatten()(pi)
        pi = Dense(self.action_size, activation='softmax')(pi)

        model = Model(inputs=input_boards, outputs=[pi, v])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.args.lr))

        return model


if __name__ == '__main__':
    from GeneralFramework.chessgame.ChessGame import ChessGame
    g = ChessGame(8)
    args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'num_channels': 256,
    })
    nn = ChessNNet(g, args)

    nn.model.save_weights('./best.h5')
