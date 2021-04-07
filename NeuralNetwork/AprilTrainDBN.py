import sys

sys.path.append('../DeepYed/NeuralNetwork/')

import argparse
from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F

from Datasets.AprilAEDataset import *

from Architecture.ImprovedAutoEncoder import ImprovedAutoEncoder

from Architecture.AutoEncoder import AutoEncoder


def train(args, load=False):
    full_bitboards = np.load("NeuralNetwork\\PreprocessedData\\Bitboards\\all_bitboards.npy")

    np.random.shuffle(full_bitboards)
    train_bitboards = full_bitboards[:int(len(full_bitboards) * .8)]
    validation_bitboards = full_bitboards[int(len(full_bitboards) * .8):]
    train_dataset = AprilAEDataset(
        data=train_bitboards)

    val_dataset = AprilAEDataset(
        data=validation_bitboards)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # autoencoder = ImprovedAutoEncoder().to(device)
    autoencoder = AutoEncoder().to(device)
    optimizer = Adam(autoencoder.parameters(), lr=args.lr)
    if load:
        loaded_state = torch.load('NeuralNetwork/Checkpoints/Jdid_Reg_AE/lr_5_decay_98/april_autoencoder_45.pth.tar',
                                  map_location=lambda storage, loc: storage)
        autoencoder.load_state_dict(loaded_state["state_dict"])
        optimizer.load_state_dict(loaded_state["optimizer"])
        new_epoch = loaded_state["epoch"]

        loss_train = list(np.genfromtxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Train_batch_res.csv".format(new_epoch-1)))
        loss_val = list(np.genfromtxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Valid_batch_results.csv".format(new_epoch-1)))

        valid_epoch = list(np.genfromtxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Valid_Res_per_epoch.csv".format(new_epoch-1)))
        train_epoch = list(np.genfromtxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Train_Res_per_epoch.csv".format(new_epoch-1)))
        all_differences = list(np.genfromtxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Differences.csv".format(new_epoch-1)))

    else:
        new_epoch = 0
        loss_train, loss_val = [], []
        train_epoch, valid_epoch = [], []
        all_differences = []


    for epoch in range(new_epoch, args.epochs + 1):
        # Training loop
        autoencoder.train()
        loss_train_batch = []
        train_loss = 0
        with tqdm(total=len(train_loader), ascii=True) as training_bar:
            training_bar.set_description(f'[Training] Epoch {epoch + 1}')
            for batch_id, (input, _) in enumerate(train_loader):
                input = input.to(device)
                optimizer.zero_grad()
                decoder_output, encoder_output = autoencoder(input)
                print(decoder_output.shape)
                loss = bce_loss(decoder_output, input)
                # loss = mse_loss(decoder_output, input)
                loss.backward()
                train_loss += loss.item()

                optimizer.step()

                average_batch_loss = loss.item() / len(input)
                loss_train_batch.append(average_batch_loss)
                training_bar.set_postfix_str("Training average batch loss: {:.3f} ".format(average_batch_loss))
                training_bar.update()

            train_epoch.append(train_loss / len(train_loader.dataset))
            training_bar.set_postfix_str("Training Mean loss: {:.4f}".format(train_loss / len(train_loader.dataset)))

        # Validation loop
        loss_val_batch = []
        autoencoder.eval()
        val_loss = 0
        with tqdm(total=len(valid_loader), ascii=True) as val_bar:
            val_bar.set_description('[Validation]')
            differences = 0
            with torch.no_grad():
                for batch_id, (input, _) in enumerate(valid_loader):
                    input = input.to(device)
                    decoder_output, encoder_output = autoencoder(input)
                    pred = (decoder_output.cpu().detach().numpy() > .5).astype(int)
                    differences += float(np.sum(input.cpu().detach().numpy() != pred))
                    loss = bce_loss(decoder_output, input)
                    # loss = mse_loss(decoder_output, input)
                    val_loss += loss.item()
                    average_batch_loss = loss.item() / len(input)
                    loss_val_batch.append(average_batch_loss)
                    val_bar.set_postfix_str("Validation average batch loss: {:.3f} ".format(average_batch_loss))
                    val_bar.update()

            val_bar.set_postfix_str(
                "Validation Mean loss: {:.4f}|| Differences: {:.3f}".format(val_loss / len(valid_loader.dataset),
                                                                            differences / len(valid_loader.dataset)))
            valid_epoch.append(val_loss / len(valid_loader.dataset))
            all_differences.append(differences / len(valid_loader.dataset))
        save_checkpoint(autoencoder, optimizer, epoch, args)
        # At first the learning rate starts from 0.005 and should be multiplied by 0.98 at the end of every epoch
        for params in optimizer.param_groups:
            params['lr'] *= args.decay
        loss_train.append(loss_train_batch)
        loss_val.append(loss_val_batch)
        np.savetxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Train_batch_res.csv".format(epoch), np.array(loss_train))
        np.savetxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Valid_batch_results.csv".format(epoch), np.array(loss_val))
        np.savetxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Valid_Res_per_epoch.csv".format(epoch), np.array(valid_epoch))
        np.savetxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Train_Res_per_epoch.csv".format(epoch), np.array(train_epoch))
        np.savetxt("NeuralNetwork/Results/Jdid_Reg_AE/{}_Differences.csv".format(epoch), np.array(all_differences))


def save_checkpoint(model, optim, epoch, args):
    state = {'state_dict': model.state_dict(),
             'optimizer': optim.state_dict(),
             'epoch': epoch + 1}
    path_to_file = 'NeuralNetwork/Checkpoints/Jdid_Reg_AE/lr_{}_decay_{}'.format(int(args.lr * 1000),
                                                                                  int(args.decay * 100))
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)
    torch.save(state, os.path.join(path_to_file, 'april_autoencoder_{}.pth.tar'.format(epoch)))


def bce_loss(decoder_result, actual_data):
    binary_cross_entropy = F.binary_cross_entropy(decoder_result, actual_data.view(-1, 773), size_average=False)
    return binary_cross_entropy


def mse_loss(decoder_result, actual_data):
    mse = F.mse_loss(decoder_result, actual_data.view(-1, 773), size_average=False)
    return mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AutoEncoder Training (DBN, Pos2Vec) as nonlinear feature extractor')

    # At first the learning rate starts from 0.005 and should be multiplied by 0.98 at the end of every epoch
    parser.add_argument('--lr', type=float, default=5e-3, help='The learning rate starts from 0.005, 0.005')

    parser.add_argument('--decay', type=float, default=0.98, help='After each epoch the lr is multiplied by this '
                                                                  'decay, 0.98')

    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs to train, 200')
    parser.add_argument('--batch-size', type=int, default=128, help='The batch size of the input, 50000')

    parser.add_argument('--checkpoint_path', default='Checkpoints/top_autoencoder.pth.tar', type=str)
    parser.add_argument('--train', default=True, action='store_true')


    if not torch.cuda.is_available():
        print('CUDA is not available.')
        device = torch.device("cpu")
    else:
        print('CUDA Available, Running on GPU!')
        device = torch.device("cuda")

    args = parser.parse_args()
    load = False
    if args.train:
        train(args, load)


