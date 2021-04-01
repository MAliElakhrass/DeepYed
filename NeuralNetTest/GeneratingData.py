import chess
import chess.pgn
import numpy as np
import random

dataPath = "./data/games_data.pgn"

num_white_moves = 1000000
num_black_moves = 1000000
num_white_moves_per_arr = 100000
num_black_moves_per_arr = 100000


def get_valid_moves(game):
    valid_moves = []

    for i, move in enumerate(game.mainline_moves()):
        if not game.board().is_capture(move) and i >= 5:
            # Append the move index to the valid_moves list
            valid_moves.append(i)

    return valid_moves


# Get bit representation of chess board
def get_bitboard(board):
    bitboard = np.zeros(2 * 6 * 64 + 5, dtype='float32')

    piece_indices = {
        'p': 0,
        'n': 1,
        'b': 2,
        'r': 3,
        'q': 4,
        'k': 5}

    for i in range(64):
        if board.piece_at(i):
            color = int(board.piece_at(i).color)
            bitboard[(6 * color + piece_indices[board.piece_at(i).symbol().lower()] + 12 * i)] = 1

    bitboard[-1] = int(board.turn)
    bitboard[-2] = int(board.has_kingside_castling_rights(True))
    bitboard[-3] = int(board.has_kingside_castling_rights(False))
    bitboard[-4] = int(board.has_queenside_castling_rights(True))
    bitboard[-5] = int(board.has_queenside_castling_rights(False))

    return bitboard


# Adds 10 moves from game to move_array at location move_index
def add_moves(game, move_array, move_index):
    valid_moves = get_valid_moves(game)
    moves_count = 0

    selected_moves = []
    for i in range(8):
        if not valid_moves:
            break

        move = random.choice(valid_moves)
        valid_moves.remove(move)
        selected_moves.append(move)
        moves_count = moves_count + 1

    board = chess.Board()
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)

        if move_index >= move_array.shape[0]:
            break

        if i in selected_moves:
            move_array[move_index] = get_bitboard(board)
            move_index += 1

    return move_index, moves_count


def iterate_over_data():
    white_moves = np.zeros((num_white_moves_per_arr, 2 * 6 * 64 + 5), dtype='float32')
    black_moves = np.zeros((num_black_moves_per_arr, 2 * 6 * 64 + 5), dtype='float32')

    # _white and black move counts store how many white and black moves have been stored
    white_move_index = 0
    black_move_index = 0
    black_moves_count = 0
    white_moves_count = 0
    count = 0
    white_count = 1
    black_count = 1
    white_empty = True
    black_empty = True

    pgn = open(dataPath)

    while True:
        # Debug printing
        if count % 1000 == 0:
            print("Game Number: {count}\twhite moves: {white_moves}\tblack moves: {black_moves}".format(
                count=count,
                black_moves=black_moves_count,
                white_moves=white_moves_count))
        game = chess.pgn.read_game(pgn)

        if not game or white_moves_count >= num_white_moves and black_moves_count >= num_black_moves:
            break
        if game.headers["Result"] == "1-0" and white_moves_count < num_white_moves:
            white_move_index, moves_count = add_moves(game, white_moves, white_move_index % num_white_moves_per_arr)
            white_moves_count = white_moves_count + moves_count
        if game.headers["Result"] == "0-1" and black_moves_count < num_black_moves:
            black_move_index, moves_count = add_moves(game, black_moves, black_move_index % num_black_moves_per_arr)
            black_moves_count = black_moves_count + moves_count

        if white_moves_count > num_white_moves_per_arr:
            print(len(white_moves))
            w_str = str(white_count)
            print("Saving white" + w_str + " array")
            np.save('data4/white' + w_str + '.npy', white_moves[:num_white_moves_per_arr])
            white_count = white_count + 1
            white_moves = np.zeros((num_white_moves_per_arr, 2 * 6 * 64 + 5), dtype='float32')
            white_move_index = 0
            white_moves_count = 0

        if black_moves_count > num_black_moves_per_arr:
            b_str = str(black_count)
            print("Saving black" + b_str + " array")
            np.save('data4/black' + b_str + '.npy', black_moves[:num_black_moves_per_arr])
            black_count = black_count + 1
            black_moves = np.zeros((num_black_moves_per_arr, 2 * 6 * 64 + 5), dtype='float32')
            black_moves_count = 0
            black_move_index = 0

        count += 1


if __name__ == '__main__':
    iterate_over_data()
