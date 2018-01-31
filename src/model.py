#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class CRNN(nn.Module):
    """LeNet5 + Recurrent Neural Network."""
    def __init__(self, input_size, batch_size, rnn_hidden, num_classes):
        super().__init__()
        self.n = int((((input_size[0]-4)/2)-4)/2)
        self.batch_size = batch_size
        self.rnn_hidden = rnn_hidden
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * self.n * self.n, 120)
        self.fc2 = nn.Linear(120, 84)
        self.lstm = nn.LSTM(self.fc2.out_features, rnn_hidden, 1)
        self.linear = nn.Linear(rnn_hidden, num_classes)

    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.rnn_hidden)
        cell = torch.zeros(1, self.batch_size, self.rnn_hidden)
        return hidden, cell

    def forward(self, x, hn):
        # pass through CNN
        feats = F.max_pool2d(F.relu(self.conv1(x)), 2)
        feats = F.max_pool2d(F.relu(self.conv2(feats)), 2)
        feats = feats.view(-1, 16 * self.n * self.n)
        feats = F.relu(self.fc1(feats))
        feats = self.fc2(feats)
        # reformat to [1 x batchSize x numFeats]
        feats = feats.unsqueeze(0)
        # pass through RNN
        outputs, hn = self.lstm(feats, hn)
        outputs = self.linear(outputs)
        return outputs, hn

def main():
    # set random seed
    torch.manual_seed(1)
    
    # hyper-parameters
    GPU = True
    batch_size = 5
    rnn_hidden = 128
    num_classes = 4
    input_size = (100,100)

    # create inputs
    inputs = torch.randn(100, batch_size, *input_size, 3)
    # reshape into [numSeqs, batchSize, numChannels, Height, Width]
    inputs = inputs.transpose(2,4)
    # store in Variables
    if GPU:
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)

    print('inputs:', inputs.size())
    
    # create CRNN object
    crnn = CRNN(input_size, batch_size, rnn_hidden, num_classes)
    if GPU:
        crnn = crnn.cuda()

    # initializeh hidden layer for RNN
    hidden, cell = crnn.init_hidden()
    if GPU:
        hidden, cell = Variable(hidden.cuda()), Variable(cell.cuda())
    else:
        hidden, cell = Variable(hidden), Variable(cell)

    print('hidden:', hidden.size())
    print('cell:', cell.size())

    # store in tuple
    hn = (hidden, cell)
    
    # for each input in sequence
    for i, inp in enumerate(inputs):
        # pass through CRNN
        outputs, hn = crnn.forward(inp, hn)

    print('outputs:', outputs.data[0])

if __name__ == '__main__':
    main()


