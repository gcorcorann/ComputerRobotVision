#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

class CRNN(nn.Module):
    """VGGNet + LSTM"""
    def __init__(self, batch_size, lstm_hidden, seq_length):
        """
        Initialize Parameters.
        
        @param  batch_size: size of batch
        @param  lstm_hidden: size of LSTM hidden layer
        """
        super().__init__()
        self.batch_size = batch_size
        self.lstm_hidden = lstm_hidden
        self.seq_length = seq_length
        # create VGG model
        self.cnn = models.vgg11_bn()
        # remove last layer
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
                )
        # lstm layer
        self.lstm = nn.LSTM(self.cnn.classifier[3].out_features, 
                lstm_hidden, 1)
        #self.linear = nn.Linear(lstm_hidden, 4)
        
    def forward(self, inputs):
        # list to hold features
        feats = []
        # for each input in sequence
        for i, inp in enumerate(inputs):
            print(i, 'inp:', inp.size())
            # pass through CNN
            out = self.cnn.forward(inp)
            feats.append(out.data)

        feats = torch.cat(feats).view(self.seq_length, self.batch_size, -1)
        feats = Variable(feats)
        print('feats:', feats.size())

        outputs, _ = self.lstm(feats)
        return outputs[-1]

    def init_hidden(self):
        hidden = torch.zeros(1, self.batch_size, self.lstm_hidden)
        cell = torch.zeros(1, self.batch_size, self.lstm_hidden)
        return hidden, cell


def main():
    """Main Function."""
    # set random seed for reproducibility
    torch.manual_seed(1)

    # hyper-parameters
    batch_size = 2
    lstm_hidden = 4
    seq_length = 5

    # create model object
    net = CRNN(batch_size, lstm_hidden, seq_length)
    print(net)

    # create inputs
    inputs = torch.randn(seq_length, batch_size, 3, 224, 224)
    # store in Variables
    inputs = Variable(inputs)
    print('inputs:', inputs.size())

    # pass through network
    output = net.forward(inputs)
    print('output:', output.size())
    print(output)


if __name__ == '__main__':
    main()
