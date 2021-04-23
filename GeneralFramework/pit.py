from GeneralFramework.Arena import Arena
from GeneralFramework.MCTS import MCTS
from GeneralFramework.chessgame.ChessGame import ChessGame
from GeneralFramework.chessgame.ChessPlayers import *
from GeneralFramework.chessgame.NNet import NNetWrapper as NNet
from GeneralFramework.utils import *
import numpy as np


if __name__ == '__main__':
    g = ChessGame(8)

    # Players
    rp = RandomPlayer(g).play
    stockish_player = StockfishPlayer(g, 1).play

    nn = NNet(g)
    nn.load_checkpoint(folder='GeneralFramework/training', filename='best.h5')

    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, nn, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    arena = Arena(n1p, stockish_player, g)

    print(arena.playGames(2, verbose=False))
