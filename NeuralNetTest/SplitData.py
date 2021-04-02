import numpy as np


if __name__ == '__main__':
    num_of_positions = 2400000
    num_of_positions_per_file = 100000
    num_of_train_positions = int(num_of_positions-100000)
    whites = np.zeros((num_of_positions, 773), dtype='float32')
    blacks = np.zeros((num_of_positions, 773), dtype='float32')
    for i in range(24):
        print(i + 1)
        whites[i * 100000:(i + 1) * 100000] = np.load('./data4/white' + str(i + 1) + '.npy')
        blacks[i * 100000:(i + 1) * 100000] = np.load('./data4/black' + str(i + 1) + '.npy')

    print("Shuffling white positions")
    np.random.shuffle(whites)
    print("Shuffling black positions")
    np.random.shuffle(blacks)

    train_whites = whites[:num_of_train_positions]
    train_blacks = blacks[:num_of_train_positions]

    val_whites = whites[num_of_train_positions:]
    val_blacks = blacks[num_of_train_positions:]

    for i in range(int(num_of_train_positions/num_of_positions_per_file)):
        np.save('./data5/white_train' + str(i+1) + '.npy', train_whites[i*num_of_positions_per_file:(i+1)*num_of_positions_per_file])
        np.save('./data5/black_train' + str(i+1) + '.npy', train_blacks[i*num_of_positions_per_file:(i+1)*num_of_positions_per_file])
        if i < int((num_of_positions-num_of_train_positions)/num_of_positions_per_file):
            np.save('./data5/white_val' + str(i + 1) + '.npy', val_whites[i * num_of_positions_per_file:(i + 1) * num_of_positions_per_file])
            np.save('./data5/black_val' + str(i + 1) + '.npy', val_blacks[i * num_of_positions_per_file:(i + 1) * num_of_positions_per_file])
