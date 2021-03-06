#! /bin/bash

import sys
sys.path.append("../DeepYed/")
from ReinforcementLearning.Arena import Arena
from ReinforcementLearning.chessgame.ChessGame import ChessGame
from ReinforcementLearning.chessgame.ChessPlayers import *
from ReinforcementLearning.chessgame.keras.NNet import NNetWrapper as NNet
from ReinforcementLearning.MCTS import MCTS
from ReinforcementLearning.utils import *
import numpy as np
import sys


if __name__ == '__main__':
    if len(sys.argv) == 3:
        level = int(sys.argv[1])
        number_games = int(sys.argv[2])
    else:
        level = 1
        number_games = 10
    # Set up
    g = ChessGame(8)
    stockish_player = StockfishPlayer(g, level).play
    nn = NNet(g)
    nn.load_checkpoint(folder='ReinforcementLearning/training', filename='best.h5')

    args1 = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})
    mcts1 = MCTS(g, nn, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    arena = Arena(n1p, stockish_player, g)

    print(arena.playGames(number_games, verbose=False))
