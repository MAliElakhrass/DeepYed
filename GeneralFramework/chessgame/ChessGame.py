from GeneralFramework.Game import Game
import chess
import numpy as np

# CONSTANTS
NUMBER_SQUARES = 8


def to_np(board):
    a = [0] * (8 * 8 * 6)
    for sq, pc in board.piece_map().items():
        a[(pc.piece_type - 1) * 64 + sq] = 1 if pc.color else -1
    return np.array(a)


class ChessGame(Game):
    def __init__(self, n):
        super().__init__()
        self.n = n or NUMBER_SQUARES

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return chess.Board()

    def toArray(self, board):
        return to_np(board)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return 6, self.n, self.n

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return (self.n ** 2) * (self.n ** 2)

    def get_uci_move(self, action, player):
        source = action // (self.n * self.n)
        target = action % (self.n * self.n)

        if player == -1:
            return chess.Move(chess.square_mirror(source), chess.square_mirror(target))
        return chess.Move(source, target)

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
        # Check if right player
        turn = 1 if board.turn else -1
        assert player == turn

        uci_move = self.get_uci_move(action, player)

        if uci_move not in board.legal_moves:
            uci_move = chess.Move.from_uci(str(uci_move) + 'q')
            if uci_move not in board.legal_moves:
                print('Error! Move is not a legal move!')

        new_board = board.copy()
        new_board.push(uci_move)

        return new_board, -player

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
        # Check if right player
        turn = 1 if board.turn else -1
        assert player == turn

        valid_moves = [0] * self.getActionSize()
        b: chess.Board = board.copy()
        for move in b.legal_moves:
            idx = move.from_square * 64 + move.to_square
            valid_moves[idx] = 1

        return np.array(valid_moves)

    def getGameEnded(self, board: chess.Board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        new_board = board.copy()

        if new_board.is_game_over(claim_draw=True):
            if new_board.result() == "1-0":
                print('Game Won!')
                return 1
            elif new_board.result() == "1/2-1/2":
                print('Draw!')
                return -0.5
            else:
                print('Game lost!')
                return -1

        return 0

    def getCanonicalForm(self, board: chess.Board, player):
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
        # Check if right player
        turn = 1 if board.turn else -1
        assert player == turn

        if board.turn:
            return board
        else:
            return board.mirror()

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
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.fen()

    def display(self, board):
        print(board)

# https://github.com/saurabhk7/chess-alpha-zero
