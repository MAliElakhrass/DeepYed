import torch
from Architecture.ImprovedSiamese import ImprovedSiamese
from torch.utils.data import DataLoader
from Datasets.SiameseDataset import *
from torch.optim import Adam
import numpy as np

def get_acc(valid_loader, model):
    e = enumerate(valid_loader)
    correct = 0
    for batch_idx, (data, label) in e:
        pred = model(data.to(device))
        correct += np.sum((pred > .5).cpu().detach().numpy() * label.numpy())
    return correct / float(valid_loader.dataset.n_inputs)



if __name__ == "__main__":
    if not torch.cuda.is_available():
        print('CUDA is not available.')
        device = torch.device("cpu")
    else:
        print('CUDA Available, Running on GPU!')
        device = torch.device("cuda")
    val_dataset = SiameseDataset(
        mode="val", n_inputs=3733410)

    train_dataset = SiameseDataset(
        mode="train", n_inputs=14917442)
    train_loader = DataLoader(train_dataset, batch_size=100000, shuffle=False)
    valid_loader = DataLoader(val_dataset, batch_size=100000, shuffle=False)

    siamese_net = ImprovedSiamese().to(device)
    state = torch.load('NeuralNetwork/Checkpoints/ImprovedSiamese/lr_10_decay_99/improved_siamese_48.pth.tar', map_location=lambda storage, loc: storage)
    siamese_net = ImprovedSiamese().to(device)
    optimizer = Adam(siamese_net.parameters(), lr=0.01)
    siamese_net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    print(get_acc(valid_loader, siamese_net))


