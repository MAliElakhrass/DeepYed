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

from AutoEncoderDataset import *
from Architecture.AutoEncoder import AutoEncoder




def train(args):
    train_dataset = AutoEncoderDataset(
        mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    autoencoder = AutoEncoder().to(device)
    optimizer = Adam(autoencoder.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        # Training loop
        autoencoder.train()
        loss_train_batch = []
        with tqdm(total=len(train_loader), ascii=True) as training_bar:
            training_bar.set_description(f'[Training] Epoch {epoch + 1}')
            training_loss = 0
            for batch_id, (input, _) in enumerate(train_loader):
                input = input.to(device)
                optimizer.zero_grad()
                decoder_output, encoder_output = autoencoder(input)
                print(decoder_output.shape)
                loss = bce_loss(decoder_output, input)

                loss.backward()
                training_loss += loss.item()
                optimizer.step()

                print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, training_loss / len(train_loader.dataset)))

                training_bar.set_postfix_str("Loss: {:.3f} ".format(loss.item()))
                training_bar.update()

            training_bar.set_postfix_str("Mean loss: {:.4f}".format(np.mean(loss_train_batch)))
            # Adjust learning rate
            for params in optimizer.param_groups:
                params['lr'] *= args.decay

# def eval(args):
#     test_dataset = AutoEncoderDataset(
#         mode="test")
#
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


def save_checkpoint(model, optim, epoch, args):
    state = {'state_dict': model.state_dict(),
             'optimizer': optim.state_dict(),
             'epoch': epoch + 1}
    path_to_file = 'Checkpoints/Autoencoder/lr_{}_decay_{}'.format(int(args.lr), int(args.decay))
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    torch.save(state, os.path.join(path_to_file, 'autoencoder_{}.pth.tar'.format(epoch)))


def bce_loss(decoder_result, actual_data):
    binary_cross_entropy = F.binary_cross_entropy(decoder_result, actual_data.view(-1, 773), size_average=False)
    return binary_cross_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AutoEncoder Training (DBN, Pos2Vec) as nonlinear feature extractor')

    # At first the learning rate starts from 0.005 and should be multiplied by 0.98 at the end of every epoch
    parser.add_argument('--lr', type=float, default=0.005, help='The learning rate starts from 0.005, 0.005')

    parser.add_argument('--decay', type=float, default=0.98, help='After each epoch the lr is multiplied by this '
                                                                  'decay, 0.98')

    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train, 10')
    parser.add_argument('--batch-size', type=int, default=1, help='The batch size of the input, 1')

    parser.add_argument('--checkpoint_path', default='Checkpoints/top_autoencoder.pth.tar', type=str)
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')

    if not torch.cuda.is_available():
        print('CUDA is not available.')
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    args = parser.parse_args()

    if args.train:
        print(sys.path)
        train(args)

    if args.eval:
        eval(args)
