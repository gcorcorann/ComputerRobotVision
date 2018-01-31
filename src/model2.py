#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

class CRNN(nn.Module):
    """VGGNet + LSTM"""
    def __init__(self, rnn_hidden):
        """
        Initialize Parameters.
        
        @param  rnn_hidden: size of LSTM hidden layer
        """
        super().__init__()
        # create VGG model
        self.cnn = models.vgg11_bn()
        # remove last layer
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
                )
        # lstm layer
        self.lstm = nn.LSTM(self.cnn.classifier[3].out_features, rnn_hidden, 1)
        self.linear = nn.Linear(rnn_hidden, 4)
        
    def forward(self, x, hn):
        """Forward Pass Through Network.

        @param  x:  data in format [batchSize, numChannels, Height, Width]
        @param  hn: hidden and cell layers for LSTM
                        @pre: tuple

        @return outputs:    output data
        """
        # pass through CNN
        feats = self.cnn.forward(x)
        # reformat for LSTM [1 x batchSize x numFeats]
        feats = feats.unsqueeze(0)
        # pass through LSTM
        outputs, hn = self.lstm(feats, hn)
        outputs = self.linear(outputs)
        return outputs

def main():
    """Main Function."""
    # hyper-parameters
    batch_size = 5

    # create model object
    net = CRNN()
    print(net)

    # create inputs
    inputs = torch.randn(batch_size, 3, 224, 224)
    # store in Variables
    inputs = Variable(inputs)
    print('inputs:', inputs.size())

    # initialize 

    # pass through model
    outputs = net.forward(inputs)
    print('outputs:', outputs.size())

if __name__ == '__main__':
    main()
