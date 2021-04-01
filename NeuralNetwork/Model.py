from datetime import datetime
from NeuralNetwork.DataPreprocessing import DataPreprocessing
from stockfish import Stockfish
from tensorflow.keras import layers, models, optimizers, callbacks
import chess
import chess.engine
import chess.pgn
import numpy as np


class DeepYed:
    def __init__(self, load=False):
        if load:
            self.model = models.load_model('model.h5')
        else:
            self.model = self.build_model(32, 4)

        self.dataprep = DataPreprocessing()

    def build_model(self, conv_size, conv_depth):
        board3d = layers.Input(shape=(14, 8, 8))

        # adding the convolutional layers
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_first')(board3d)
        for _ in range(conv_depth):
            previous = x
            x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_first')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', data_format='channels_first')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, previous])
            x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1, 'sigmoid')(x)

        return models.Model(inputs=board3d, outputs=x)

    def get_dataset(self):
        container = np.load('./data/dataset.npz')
        self.X, self.Y = container['b'], container['v']
        self.Y = np.asarray(self.Y / abs(self.Y).max() / 2 + 0.5, dtype=np.float32)  # Normalization

    def train(self):
        self.model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
        self.model.summary()
        self.model.fit(self.X, self.Y,
                       batch_size=2048,
                       epochs=1000,
                       verbose=1,
                       validation_split=0.1,
                       callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                                  callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])

        self.model.save('model.h5')

    # used for the minimax algorithm
    def minimax_eval(self, board):
        board3d = self.dataprep.split_dims(board)
        board3d = np.expand_dims(board3d, 0)

        return self.model.predict(board3d)[0][0]

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.minimax_eval(board)

        if maximizing_player:
            max_eval = -np.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = np.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    # this is the actual function that gets the move from the neural network
    def get_ai_move(self, board, depth):
        max_move = None
        max_eval = -np.inf

        for move in board.legal_moves:
            board.push(move)
            eval = self.minimax(board, depth - 1, -np.inf, np.inf, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                max_move = move

        return max_move

    def play_stockfish(self):
        stockfish = Stockfish('./engines/stockfish-12/stockfish.exe',
                              parameters={"Threads": 16, "Skill Level": 1})

        game = chess.pgn.Game()
        game.headers["Event"] = "Test"
        game.headers["Site"] = "Hammad's PC"
        game.headers["Date"] = str(datetime.now().date())
        game.headers["Round"] = '1'
        game.headers["White"] = "DeepYed"
        game.headers["Black"] = 'Stockfish'

        history = []
        board = chess.Board()
        while not board.is_game_over():
            if board.turn:
                print("DeepYed's Turn")
                move = self.get_ai_move(board, depth=1)
                history.append(move)
                board.push(move)
                print(move)
            else:
                print("Stockfish's Turn")
                move = chess.Move.from_uci(self.get_move_stockfish(board, stockfish))
                history.append(move)
                board.push(move)
                print(move)

        game.add_line(history)
        game.headers["Result"] = str(board.result())

        print(game)
        print(game, file=open("round_1.pgn", "w"), end="\n\n")

    def get_move_stockfish(self, board, stockfish):
        fen = board.fen()
        stockfish.set_fen_position(fen)

        return stockfish.get_best_move_time(100)


if __name__ == '__main__':
    deep_yed = DeepYed(load=False)
    deep_yed.get_dataset()
    deep_yed.train()

    # deep_yed.play_stockfish()
