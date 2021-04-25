from stockfish import Stockfish


class StockfishPlayer:
    def __init__(self, game, level):
        self.game = game
        self.engine = Stockfish('engines/stockfish.exe', parameters={"Threads": 4, "Skill Level": level})
        self.letters = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}

    def play(self, board):
        fen = board.fen()
        self.engine.set_fen_position(fen)

        move = self.engine.get_best_move_time(100)

        x1 = self.letters[move[0]] - 1
        x2 = self.letters[move[2]] - 1
        y1 = int(move[1]) - 1
        y2 = int(move[3]) - 1

        return (8 * 8) * (8 * x1 + y1) + (8 * x2 + y2)
