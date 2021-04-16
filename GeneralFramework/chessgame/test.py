import chess
import numpy


def bb2array(b):  # board to vector of len 64
    x = numpy.zeros(64, dtype=numpy.int8)
    # print('Flipping: ', flip)
    for pos in range(64):
        piece = b.piece_type_at(pos)  # Gets the piece type at the given square. 0==>blank,1,2,3,4,5,6
        if piece:
            color = int(bool(b.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))  # to check if piece is black or white
            # print ('piece: ', piece, 'b.occupied_co[chessg.BLACK]: ', b.occupied_co[chessg.BLACK], 'chessg.BB_SQUARES[pos]: ', chessg.BB_SQUARES[pos], 'color: ', color, 'pos: ', pos, '\t', b.occupied_co[chessg.BLACK] & chessg.BB_SQUARES[pos])
            col = int(pos % 8)
            row = int(pos / 8)
            #		if flip:
            #		row = 7-row
            #		color = 1 - color
            x[row * 8 + col] = -piece if color else piece
    t = b.turn
    c = b.castling_rights
    e = b.ep_square
    h = b.halfmove_clock
    f = b.fullmove_number
    return numpy.reshape(x, (8,8))


def get_bitboard(board):
    """
    Convert a board to numpy array of size 8x8
    :param board:
    :return: numpy array 8*8
    """
    bitboard = numpy.zeros([8, 8]).astype(str)

    for square in range(8 * 8):
        if board.piece_at(square):
            piece = board.piece_at(square).symbol()
        else:
            piece = ' '
        col = int(square % 8) - 8
        row = int(square / 8) - 8
        bitboard[row][col] = piece

    return bitboard


if __name__ == '__main__':
    b = chess.Board()
    print(bb2array(b))
    print(get_bitboard(b))
