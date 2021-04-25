"""
This code was taken from : https://stackoverflow.com/a/61439875/9319274
"""
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QWidget
import chess
import chess.svg


class MainWindow(QWidget):
    def __init__(self, board):
        super().__init__()

        self.setGeometry(100, 100, 520, 520)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 500, 500)

        self.set_board(board)

    def set_board(self, board):
        chessboardSvg = chess.svg.board(board).encode("UTF-8")
        self.widgetSvg.load(chessboardSvg)
