from PyQt5.QtWidgets import QApplication
from Heuristic.MainWindow import MainWindow
from stockfish import Stockfish
import chess
import chess.svg
import chess.pgn
import chess.polyglot
import chess.engine
import datetime

# Piece Square Tables
# Value from https://github.com/thomasahle/sunfish
pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    -31,   8,  -7, -37, -36, -14,   3, -31,
    -22,   9,   5, -11, -10,  -2,   3, -19,
    -26, 3, 10, 9, 6, 1, 0, -23,
    -17,  16,  -2,  15,  14,   0,  15, -13,
    7,  29,  21,  44,  40,  31,  44,   7,
    78,  83,  86,  73, 102,  82,  85,  90,
    0, 0, 0, 0, 0, 0, 0, 0]

knightstable = [
    -74, -23, -26, -24, -19, -35, -22, -69,
    -23, -15,   2,   0,   2,   0, -23, -20,
    -18,  10,  13,  22,  18,  15,  11, -14,
    -1,   5,  31,  21,  22,  35,   2,   0,
    24,  24,  45,  37,  33,  41,  25,  17,
    10,  67,   1,  74,  73,  27,  62,  -2,
    -3,  -6, 100, -36,   4,  62,  -4, -14,
    -66, -53, -75, -75, -10, -55, -58, -70]

bishopstable = [
    -7,   2, -15, -12, -14, -15, -10, -10,
    19,  20,  11,   6,   7,   6,  20,  16,
    14,  25,  24,  15,   8,  25,  20,  15,
    13,  10,  17,  23,  17,  16,   0,   7,
    25,  17,  20,  34,  26,  25,  15,  10,
    -9,  39, -32,  41,  52, -10,  28, -14,
    -11,  20,  35, -42, -39,  31,   2, -22,
    -59, -78, -82, -76, -23,-107, -37, -50]

rookstable = [
    -30, -24, -18,   5,  -2, -18, -31, -32,
    -53, -38, -31, -26, -29, -43, -44, -53,
    -42, -28, -42, -25, -25, -35, -26, -46,
    -28, -35, -16, -21, -13, -29, -46, -30,
    0,   5,  16,  13,  18,  -4,  -9,  -6,
    19,  35,  28,  33,  45,  27,  25,  15,
    55,  29,  56,  67,  55,  62,  34,  60,
    35,  29,  33,   4,  37,  33,  56,  50]

queenstable = [
    -39, -30, -31, -13, -31, -36, -34, -42,
    -36, -18,   0, -19, -15, -15, -21, -38,
    -30,  -6, -13, -11, -16, -11, -16, -27,
    -14, -15,  -2,  -5,  -1, -10, -20, -22,
    1, -16,  22,  17,  25,  20, -13,  -6,
    -2,  43,  32,  60,  72,  63,  43,   2,
    14,  32,  60, -10,  20,  76,  57,  24,
    6,   1,  -8,-104,  69,  24,  88,  26]

kingstable = [
    17,  30,  -3, -14,   6,  -1,  40,  18,
    -4,   3, -14, -50, -57, -18,  13,   4,
    -47, -42, -43, -79, -64, -32, -29, -32,
    -55, -43, -52, -28, -51, -47,  -8, -50,
    -55,  50,  11,  -4, -19,  13,   0, -49,
    -62,  12, -57,  44, -67,  28,  37, -31,
    -32,  10,  55,  56,  56,  55,  10,   3,
    4,  54,  47, -99, -99,  60,  83, -62]


class PlayChess:
    def __init__(self):
        self.history = []
        self.board = chess.Board()

    def play_engine(self, level=1, number_games=10):
        """
        This function will evaluate our agent against Stockfish

        :param number_games: The number of games to be played against Stockfish
        :param level: Level of Stockfish
        :return:
        """
        stockfish = Stockfish('engines/stockfish-12/stockfish.exe', parameters={"Threads": 4, "Skill Level": level})

        for i in range(number_games):
            game = chess.pgn.Game()
            game.headers["Event"] = "Evalutation DeepYed vs Stockfish"
            game.headers["Site"] = "My PC"
            game.headers["Date"] = str(datetime.datetime.now().date())
            game.headers["Round"] = i
            game.headers["White"] = "DeepYed"
            game.headers["Black"] = 'Stockfish'

            while not self.board.is_game_over():
                if self.board.turn:
                    print("DeepYed's Turn")
                    move = self.select_move(depth=3)
                    self.history.append(move)
                    self.board.push(move)
                    print(move)
                else:
                    print("Stockfish's Turn")
                    move = chess.Move.from_uci(self.play_stockfish(stockfish))
                    self.history.append(move)
                    self.board.push(move)
                    print(move)

            game.add_line(self.history)
            game.headers["Result"] = str(self.board.result())

            print(game)
            print(game, file=open(f"round_{i}.pgn", "w"), end="\n\n")

        self.show_board()

    def play_stockfish(self, stockfish):
        """
        This function will return the best move chosen by the stockfish engine in 1 second

        :param stockfish: The stockfish engine
        :return: The uci move
        """
        fen = self.board.fen()
        stockfish.set_fen_position(fen)

        return stockfish.get_best_move_time(1000)

    def evaluate_board(self):
        """
        Heuristic. The heuristic is calculated by adding the material pieces and the positions of each piece

        :return: the heuristic value
        """
        if self.board.is_checkmate():
            if self.board.turn:
                return -9999
            else:
                return 9999
        if self.board.is_stalemate():
            return 0
        if self.board.is_insufficient_material():
            return 0

        wp = len(self.board.pieces(chess.PAWN, chess.WHITE))
        bp = len(self.board.pieces(chess.PAWN, chess.BLACK))
        wn = len(self.board.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(self.board.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(self.board.pieces(chess.BISHOP, chess.WHITE))
        bb = len(self.board.pieces(chess.BISHOP, chess.BLACK))
        wr = len(self.board.pieces(chess.ROOK, chess.WHITE))
        br = len(self.board.pieces(chess.ROOK, chess.BLACK))
        wq = len(self.board.pieces(chess.QUEEN, chess.WHITE))
        bq = len(self.board.pieces(chess.QUEEN, chess.BLACK))

        material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)

        pawnsq = sum([pawntable[i] for i in self.board.pieces(chess.PAWN, chess.WHITE)])
        pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)] for i in self.board.pieces(chess.PAWN, chess.BLACK)])
        knightsq = sum([knightstable[i] for i in self.board.pieces(chess.KNIGHT, chess.WHITE)])
        knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)] for i in self.board.pieces(chess.KNIGHT, chess.BLACK)])
        bishopsq = sum([bishopstable[i] for i in self.board.pieces(chess.BISHOP, chess.WHITE)])
        bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)] for i in self.board.pieces(chess.BISHOP, chess.BLACK)])
        rooksq = sum([rookstable[i] for i in self.board.pieces(chess.ROOK, chess.WHITE)])
        rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)] for i in self.board.pieces(chess.ROOK, chess.BLACK)])
        queensq = sum([queenstable[i] for i in self.board.pieces(chess.QUEEN, chess.WHITE)])
        queensq = queensq + sum([-queenstable[chess.square_mirror(i)] for i in self.board.pieces(chess.QUEEN, chess.BLACK)])
        kingsq = sum([kingstable[i] for i in self.board.pieces(chess.KING, chess.WHITE)])
        kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)] for i in self.board.pieces(chess.KING, chess.BLACK)])

        heuristic = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
        if self.board.turn:
            return heuristic
        else:
            return -heuristic

    def quiesce(self, alpha, beta):
        """
        This function will perform a quiescence search.

        :param alpha:
        :param beta:
        :return:
        """
        stand_pat = self.evaluate_board()

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        for move in self.board.legal_moves:
            if self.board.is_capture(move):
                self.board.push(move)
                score = -self.quiesce(-beta, -alpha)
                self.board.pop()

                if score >= beta:
                    return beta
                elif score > alpha:
                    alpha = score

        return alpha

    def alphabeta(self, alpha, beta, depth):
        """
        Negamax search algorithm

        :param alpha:
        :param beta:
        :param depth:
        :return:
        """
        if depth == 0:
            return self.quiesce(alpha, beta)

        for move in self.board.legal_moves:
            self.board.push(move)
            score = -self.alphabeta(-beta, -alpha, depth-1)
            self.board.pop()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha

    def select_move(self, depth):
        """
        Select a move for the agent. At first, try to obtain a move from the opening book

        :param depth:
        :return:
        """
        try:
            return chess.polyglot.MemoryMappedReader('books/Perfect2017-SF12.bin')\
                .weighted_choice(board=self.board).move
        except:
            best_move = chess.Move.null()
            best_value = -99999
            alpha = -100000
            beta = 100000
            for move in self.board.legal_moves:
                self.board.push(move)
                board_value = -self.alphabeta(-beta, -alpha, depth-1)

                if board_value > best_value:
                    best_value = board_value
                    best_move = move
                if board_value > alpha:
                    alpha = board_value

                self.board.pop()

            # Return random move
            if best_move == chess.Move.null():
                legal_moves_iterator = self.board.generate_legal_moves()
                best_move = next(x for x in legal_moves_iterator)

            return best_move

    def show_board(self):
        """
        Visual representation of the chess board

        :return:
        """
        app = QApplication([])
        window = MainWindow(self.board)
        window.show()
        app.exec()


if __name__ == "__main__":
    player = PlayChess()
    player.play_engine()
