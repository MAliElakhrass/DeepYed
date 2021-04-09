from torch import Tensor
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels=13, num_actions=8064):  # 4032*2
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.num_actions = num_actions
        self.out_channels1 = 64
        self.out_channels2 = 128
        self.out_channels3 = 256

        self.conv1 = nn.Conv2d(in_channels, self.out_channels1, 3)
        self.bn1 = nn.BatchNorm2d(self.out_channels1)
        self.conv2 = nn.Conv2d(self.out_channels1, self.out_channels2, 3)
        self.bn2 = nn.BatchNorm2d(self.out_channels2)
        self.conv3 = nn.Conv2d(self.out_channels2, self.out_channels3, 3)
        self.bn3 = nn.BatchNorm2d(self.out_channels3)

        self.wide = self.feature_size(self.feature_size(self.feature_size(8, 3), 3), 3)
        # wide = 8 - 2 - 2 - 2 = 2

        self.in_fc1 = self.out_channels3 * self.wide * self.wide  # 256 * 4 = 1024
        self.in_fc2 = 4096
        self.in_fc3 = 4096

        # policy layer: propability over action space
        self.fc1_pol = nn.Linear(self.in_fc1, self.in_fc2)
        self.fc2_pol = nn.Linear(self.in_fc2, self.in_fc3)
        self.fc3_pol = nn.Linear(self.in_fc3, num_actions)

        # value layer: value for current state
        self.fc1_val = nn.Linear(self.in_fc1, 128)
        self.fc2_val = nn.Linear(128, 16)
        self.fc3_val = nn.Linear(16, 1)

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = output.view(-1, self.in_fc1)

        # policy layer: propability over action space
        output_policy = F.relu(self.fc1_pol(output))
        output_policy = F.relu(self.fc2_pol(output_policy))
        output_policy = F.softmax(self.fc3_pol(output_policy), dim=1)

        # value layer: value for current state
        output_value = F.relu(self.fc1_val(output))
        output_value = F.relu(self.fc2_val(output_value))
        output_value = self.fc3_val(output_value)

        return output_policy, output_value

    def feature_size(self, size, kernel_size, stride=1, padding=0):
        return int((size - kernel_size + 2 * padding) / stride + 1)


class NeuralNetwork:
    def __init__(self, lr, device):
        self.lr = lr
        self.device = device

        self.policy_net = CNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.best_loss = 99999999999  # initial threshold to save model
        self.L2_lambda = 0.00001

    def save_model(self, loss):
        if loss.item() < self.best_loss:
            torch.save({
                'model_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, 'best_policy.model')
            self.best_loss = loss.item()

    def load_model(self):
        checkpoint = torch.load('best_policy.model')
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def optimize_model(self, buffer):
        torch.cuda.empty_cache()

        # init
        state, probability, winner = buffer
        state_opt = Variable(Tensor(np.array(state))).to(self.device)
        probability_opt = Variable(Tensor(np.array(probability))).to(self.device)
        winner_opt = Variable(Tensor(np.array(winner))).to(self.device)

        # calculate policy network
        action_probabilities, value = self.policy_net(state_opt)

        # min error
        value_loss = F.mse_loss(value.view(-1), winner_opt)
        policy_loss = -torch.mean(torch.sum(probability_opt*(torch.log(action_probabilities)), 1))
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()

        # L2 reg
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.policy_net.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, 2)
        loss += self.L2_lambda*l2_reg

        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.save_model(loss)

        return loss
