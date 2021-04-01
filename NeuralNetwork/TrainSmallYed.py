import sys

sys.path.append('../DeepYed/NeuralNetwork/')

import argparse
from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from torch.nn import functional as F

from Datasets.UpdatedSiameseDataset import *

from Architecture.SmallerSiamese import SmallerSiamese
from Architecture.SiameseNet import SiameseNet
from Architecture.ImprovedSiamese import ImprovedSiamese

def train(args):
    # The number specified here should not be bigger than the number of losses or number of wins, otherwise an index error will occur
    train_dataset = UpdatedSiameseDataset(
        mode="train",  length=4000000, chunk_size=400000)

    val_dataset = UpdatedSiameseDataset(
        mode="valid",  length=500000, chunk_size=50000)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    siamese_net = ImprovedSiamese().to(device)
    optimizer = Adam(siamese_net.parameters(), lr=args.lr)
    loss_train, loss_val = [], []
    train_epoch, valid_epoch = [], []
    val_accuracies, train_accuracies = [], []
    for epoch in range(args.epochs):

        # Training loop
        siamese_net.train()
        loss_train_batch = []
        train_loss = 0
        correct = 0
        with tqdm(total=len(train_loader), ascii=True) as training_bar:
            training_bar.set_description(f'[Training] Epoch {epoch + 1}')
            for batch_id, (input, label) in enumerate(train_loader):
                input = input.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                pred = siamese_net(input)
                correct += np.sum((pred > .5).cpu().detach().numpy() * label.cpu().numpy())
                loss = bce_loss(pred, label)
                loss.backward()
                train_loss += loss.item()

                optimizer.step()

                average_batch_loss = loss.item() / len(input)
                loss_train_batch.append(average_batch_loss)
                training_bar.set_postfix_str("Training average batch loss: {:.5f} ".format(average_batch_loss))
                training_bar.update()

            train_epoch.append(train_loss / len(train_loader.dataset))
            train_accuracies.append(correct/len(train_loader.dataset))
            training_bar.set_postfix_str("Training Mean loss: {:.5f}, Training Accuracy: {:.5f} ".format((train_loss / len(train_loader.dataset)), (correct/len(train_loader.dataset))))

        # Validation loop
        loss_val_batch = []
        siamese_net.eval()
        val_loss = 0
        correct = 0
        with tqdm(total=len(valid_loader), ascii=True) as val_bar:
            val_bar.set_description('[Validation]')
            with torch.no_grad():
                for batch_id, (input, label) in enumerate(valid_loader):
                    input = input.to(device)
                    label = label.to(device)
                    pred = siamese_net(input)
                    correct += np.sum((pred > .5).cpu().detach().numpy() * label.cpu().numpy())
                    loss = bce_loss(pred, label)
                    val_loss += loss.item()
                    average_batch_loss = loss.item() / len(input)
                    loss_val_batch.append(average_batch_loss)
                    val_bar.set_postfix_str("Validation average batch loss: {:.5f} ".format(average_batch_loss))
                    val_bar.update()

            valid_epoch.append(val_loss / len(valid_loader.dataset))
            val_accuracies.append(correct/len(valid_loader.dataset))
            val_bar.set_postfix_str("Validation Mean loss: {:.5f}, Validation Accuracy: {:.5f}".format((val_loss / len(valid_loader.dataset)), (correct/len(valid_loader.dataset))))

        save_checkpoint(siamese_net, optimizer, epoch, args)
        # At first the learning rate starts from 0.01 and should be multiplied by 0.99 at the end of every epoch
        for params in optimizer.param_groups:
            params['lr'] *= args.decay
        loss_train.append(loss_train_batch)
        loss_val.append(loss_val_batch)
        np.savetxt("NeuralNetwork/Results/New_regular_siamese/{}_Train_batch_res.csv".format(epoch), np.array(loss_train))
        np.savetxt("NeuralNetwork/Results/New_regular_siamese/{}_Valid_batch_results.csv".format(epoch), np.array(loss_val))
        np.savetxt("NeuralNetwork/Results/New_regular_siamese/{}_Valid_Loss_per_epoch.csv".format(epoch), np.array(valid_epoch))
        np.savetxt("NeuralNetwork/Results/New_regular_siamese/{}_Train_Loss_per_epoch.csv".format(epoch), np.array(train_epoch))
        np.savetxt("NeuralNetwork/Results/New_regular_siamese/{}_Valid_ACC_per_epoch.csv".format(epoch), np.array(val_accuracies))
        np.savetxt("NeuralNetwork/Results/New_regular_siamese/{}_Train_ACC_per_epoch.csv".format(epoch), np.array(train_accuracies))


def save_checkpoint(model, optim, epoch, args):
    state = {'state_dict': model.state_dict(),
             'optimizer': optim.state_dict(),
             'epoch': epoch + 1}
    path_to_file = 'NeuralNetwork/Checkpoints/New_regular_siamese/lr_{}_decay_{}'.format(int(args.lr * 1000),
                                                                                int(args.decay * 100))
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)
    torch.save(state, os.path.join(path_to_file, 'new_reg_siamese_{}.pth.tar'.format(epoch)))


def bce_loss(pred, label):
    binary_cross_entropy = F.binary_cross_entropy(pred, label, size_average=False)
    return binary_cross_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Siamese Network Training (DeepYed)')

    # At first the learning rate starts from 0.01 and should be multiplied by 0.99 at the end of every epoch
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate starts from 0.01, 0.01')

    parser.add_argument('--decay', type=float, default=0.99, help='After each epoch the lr is multiplied by this '
                                                                  'decay, 0.99')

    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs to train, 1000')
    parser.add_argument('--batch-size', type=int, default=128, help='The batch size of the input, 100000')

    parser.add_argument('--checkpoint_path', default='Checkpoints/top_autoencoder.pth.tar', type=str)
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')

    if not torch.cuda.is_available():
        print('CUDA is not available.')
        device = torch.device("cpu")
    else:
        print('CUDA Available, Running on GPU!')
        device = torch.device("cuda")

    args = parser.parse_args()

    if args.train:
        train(args)

    if args.eval:
        eval(args)
