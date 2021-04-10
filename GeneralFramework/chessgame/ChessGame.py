from GeneralFramework.Game import Game
from GeneralFramework.chessgame.ChessLogic import Board
import numpy as np


# CONSTANTS
NUMBER_SQUARES = 8


class ChessGame(Game):
    def __init__(self, n):
        super().__init__()
        self.n = n or NUMBER_SQUARES
        self.dict_number_letter = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h'}

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        b = Board(self.n)
        return b.board

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.n, self.n

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return (self.n**2)*(self.n**2)

    def get_uci_move(self, x1, y1, x2, y2):
        row1 = self.dict_number_letter[x1 + 1]
        row2 = self.dict_number_letter[x2 + 1]
        y1 += 1
        y2 += 1

        return row1 + str(y1) + row2 + str(y2)

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # For debug purpose
        print('ChessGame==>getNextState, param action number: ', action)
        
        # If no possible action
        if action == NUMBER_SQUARES * NUMBER_SQUARES:
            return board, -player

        new_board = Board(self.n)
        new_board.board = board.copy() # Copy the chessboard
        temp1 = int(action / (NUMBER_SQUARES * NUMBER_SQUARES))
        temp2 = action % (NUMBER_SQUARES * NUMBER_SQUARES)
        x1 = int(temp1 / NUMBER_SQUARES)
        y1 = temp1 % NUMBER_SQUARES
        x2 = int(temp2 / NUMBER_SQUARES)
        y2 = temp2 % NUMBER_SQUARES
        print("selected action: ", x1, y1, x2, y2)

        move = self.get_uci_move(x1, y1, x2, y2)

        print("MOVE: ", move)

        new_board.make_move(move)

        return new_board.board, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valid_moves = [0] * self.getActionSize()
        b = Board(self.n)
        b.board = board.copy()
        legal_moves = b.get_valid_moves()

        if len(legal_moves) == 0:
            valid_moves[-1] = 1
            return np.array(valid_moves)

        for move in legal_moves:
            x1 = move[0]
            y1 = move[1]
            x2 = move[2]
            y2 = move[3]

            valid_moves[(self.n * self.n) * (self.n * x1 + y1) + (self.n * x2 + y2)] = 1

        return np.array(valid_moves)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        pass

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chessgame,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass


# https://github.com/saurabhk7/chess-alpha-zero
