from PyQt5.QtWidgets import QApplication
from Projet.Simple.MainWindow import MainWindow
import chess
import chess.svg
import chess.pgn
import chess.polyglot
import chess.engine
import datetime

# Piece Square Tables
# Value from https://www.chessprogramming.org/Simplified_Evaluation_Function
pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]

bishopstable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]

rookstable = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0]

queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]

kingstable = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30]


class PlayChess():
    def __init__(self):
        self.board = chess.Board()
        self.history = []

    def start_game(self):
        while True:
            # AI
            move = self.select_move(depth=3)
            self.board.push(move)
            self.show_board()
            try:
                user_move = None
                while user_move not in self.board.legal_moves:
                    user_input = input("Entre un move")
                    user_move = chess.Move.from_uci(user_input)
                self.board.push(user_move)
            except:
                print('HMAR')
                user_input = input("Entre un move")
                user_move = chess.Move.from_uci(user_input)
                self.board.push(user_move)
            # self.board.push_san(user_input)

    def evalute_stockfish(self):
        engine = chess.engine.SimpleEngine.popen_uci('./../stockfish-12/stockfish_20090216_x64_bmi2.exe')
        print("Engine Loaded")

        game = chess.pgn.Game()
        game.headers["Event"] = "MATCH HISTORIQUE>"
        game.headers["Site"] = "Hammad's PC"
        game.headers["Date"] = str(datetime.datetime.now().date())
        game.headers["Round"] = '1'
        game.headers["White"] = "DeepYed"
        game.headers["Black"] = "Stockfish12"

        history = []
        counter = 1
        while not self.board.is_game_over():
            print("Move: ", counter)
            if self.board.turn:
                print('White Turn')
                move = self.select_move(depth=3)
                self.board.push(move)
                print(move)
            else:
                print('Black Turn')
                result = engine.play(self.board, chess.engine.Limit(time=1))
                history.append(result.move)
                self.board.push(result.move)
                print(result.move)

            counter += 1

        engine.close()

        game.add_line(history)
        game.headers["Result"] = str(self.board.result())

        print(game)
        print(game, file=open("resultat.pgn", "w"), end="\n\n")

        self.show_board()

    def evaluate_board(self):
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

    # https://www.chessprogramming.org/Quiescence_Search
    def quiesce(self, alpha, beta):
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

    # https://www.chessprogramming.org/Alpha-Beta
    def alphabeta(self, alpha, beta, depth):
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
        try:
            move = chess.polyglot.MemoryMappedReader('Perfect2017-SF12.bin').weighted_choice(board=self.board).move
            self.history.append(move)
            return move
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
            self.history.append(best_move)

            return best_move

    def show_board(self):
        app = QApplication([])
        window = MainWindow(self.board)
        window.show()
        app.exec()


if __name__ == "__main__":
    player = PlayChess()
    # player.start_game()
    player.evalute_stockfish()

