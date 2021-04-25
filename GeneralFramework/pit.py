from Arena import Arena
from MCTS import MCTS
from chessgame.ChessGame import ChessGame
from chessgame.ChessPlayers import *
from chessgame.keras.NNet import NNetWrapper as NNet
from utils import *
import numpy as np
import sys

if __name__ == '__main__':
    level = sys.argv[1] or 1
    number_games = sys.argv[2] or 10

    # Set up
    g = ChessGame(8)
    stockish_player = StockfishPlayer(g, level).play
    nn = NNet(g)
    nn.load_checkpoint(folder='GeneralFramework/training', filename='best.h5')

    args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts1 = MCTS(g, nn, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    arena = Arena(n1p, stockish_player, g)

    print(arena.playGames(number_games, verbose=False))
