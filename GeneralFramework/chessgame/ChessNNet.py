from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# import tensorflow as tf


class ChessNNet:
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.model = self.construct_graph()

    def construct_graph(self):
        input_boards = Input(name='input', shape=(self.board_x, self.board_y))
        dropout = Input(name='dropout')

        x_image = Reshape((self.board_x, self.board_y, 1))(input_boards)
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args['num_channels'], padding='same', use_bias=False)(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args['num_channels'], padding='same', use_bias=False)(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args['num_channels'], padding='same', use_bias=False)(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args['num_channels'], padding='same', use_bias=False)(h_conv3)))
        h_conv5 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args['num_channels'], padding='same', use_bias=False)(h_conv4)))
        h_conv5_flat = Flatten()(h_conv5)

        s_fc1 = Dropout(self.args['dropout'])(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv5_flat))))
        s_fc2 = Dropout(self.args['dropout'])(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))

        # pi = tf.keras.layers.Dense(s_fc2, self.action_size)  # batch_size x self.action_size
        # prob = tf.nn.softmax(pi)
        # v = tf.nn.tanh(tf.keras.layers.Dense(s_fc2, 1))
        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        v = Dense(1, activation='tanh', name='v')(s_fc2)

        model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.args['lr']))

        return model

    """
    def conv2d(self, x, out_channels, padding):
        return tf.nn.conv2d(x, out_channels, kernel_size=[3, 3], padding=padding)

    def calculate_loss(self):
        target_pis = tf.keras.Input(name='target_pis', shape=[None, self.action_size], dtype=tf.dtypes.float32)
        target_vs = tf.keras.Input(name='target_vs', shape=[None], dtype=tf.dtypes.float32)

        loss_pi = tf.nn.softmax_cross_entropy_with_logits(target_pis, self.prob)
        loss_v = tf.keras.losses.mean_squared_error(target_vs, tf.reshape(self.v, shape=[-1, ]))
        total_loss = loss_pi + loss_v

        train_step = tf.keras.optimizers.Adam(self.args['lr']).minimize(total_loss)
    """
