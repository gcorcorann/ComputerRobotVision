#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class LeNet5(nn.Module):
    """
    LeNet5.
    """
    def __init__(self, rnn_hidden, num_layers, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(20*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.lstm = nn.LSTM(self.fc2.out_features, rnn_hidden, num_layers)
        self.linear = nn.Linear(rnn_hidden, num_classes)

    def forward(self, inputs):
        # list to hold features
        feats = []
        # for each input in sequence
        for inp in inputs:
            # pass through CNN
            out = self.pool1(F.relu(self.conv1(inp)))
            out = self.pool2(F.relu(self.conv2(out)))
            # reshape into [batchSize x numFeats]
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            # store in feature list
            feats.append(out.data)

        # format features and store in Variable
        feats = torch.cat(feats).view(len(feats), -1, self.fc2.out_features)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.linear(outputs[-1, :, :])
        return outputs

class Network1(nn.Module):
    """
    LeNet5 + LSTM.
    """
    def __init__(self, input_size, batch_size, rnn_hidden, num_classes,
            seq_length):
        super().__init__()
        self.n = int((((input_size[0]-4)/2)-4)/2)
        self.batch_size = batch_size
        self.rnn_hidden = rnn_hidden
        self.seq_length = seq_length
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * self.n * self.n, 120)
        self.fc2 = nn.Linear(120, 84)
        self.lstm = nn.LSTM(self.fc2.out_features, rnn_hidden, 1)
        self.linear = nn.Linear(rnn_hidden, num_classes)

    def forward(self, inputs):
        # list to hold features
        feats = []
        # for each input in sequence
        for i, inp in enumerate(inputs):
            # pass through CNN
            out = F.max_pool2d(F.relu(self.conv1(inp)), 2)
            out = F.max_pool2d(F.relu(self.conv2(out)), 2)
            out = out.view(-1, 16 * self.n * self.n)
            out = F.relu(self.fc1(out))
            #TODO add relu
            out = self.fc2(out)
            # store in feature list
            feats.append(out.data)

        # format features and store in Variable
        feats = torch.cat(feats).view(self.seq_length, -1, 
                self.fc2.out_features)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.linear(outputs[-1])
        return outputs

class Network2(nn.Module):
    """
    PreTrained VGGNet + LSTM.
    """
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
        self.cnn = models.vgg16_bn(pretrained=True)
        # remove last layer
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
                )
        # use CNN as feature extractor
        for param in self.cnn.parameters():
            param.requires_grad = False
        # lstm layer
        self.lstm = nn.LSTM(self.cnn.classifier[3].out_features, 
                lstm_hidden, 1)
        self.linear = nn.Linear(lstm_hidden, 4)
        
    def forward(self, inputs):
        # list to hold features
        feats = []
        # for each input in sequence
        for i, inp in enumerate(inputs):
            # pass through CNN
            out = self.cnn.forward(inp)
            feats.append(out.data)

        # format features and store in Variable
        feats = torch.cat(feats).view(self.seq_length, -1,
                self.cnn.classifier[3].out_features)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.linear(outputs[-1])
        return outputs

class Network3(nn.Module):
    """
    ResNet + LSTM.
    """
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
        # create ResNet model
        self.cnn = models.resnet18()
        # change first cnn layer for flow (i.e. 2 dimensions)
        self.cnn.conv1 = nn.Conv2d(2, 64, kernel_size=(7,7), stride=(2,2),
                padding=(3,3), bias=False)
        self.num_feats = self.cnn.fc.in_features
        # remove last layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        # use CNN as feature extractor
#        for param in self.cnn.parameters():
#            param.requires_grad = False
        # lstm layer
        self.lstm = nn.LSTM(self.num_feats, lstm_hidden, 1)
        self.linear = nn.Linear(lstm_hidden, 4)
        
    def forward(self, inputs):
        # list to hold features
        feats = []
        # for each input in sequence
        for i, inp in enumerate(inputs):
            # pass through CNN
            out = self.cnn.forward(inp)
            feats.append(out.data)

        # format features and store in Variable
        feats = torch.cat(feats).view(self.seq_length, -1, self.num_feats)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.linear(outputs[-1])
        return outputs

def main():
    import time

    # start timer
    start = time.time()
    # set random seed
    torch.manual_seed(1)
    
    # hyper-parameters
    GPU = torch.cuda.is_available()
    seq_length = 10
    batch_size = 5
    rnn_hidden = 128
    num_classes = 4
    input_size = (224,224)

    # create model object
#    net = Network1(input_size, batch_size, rnn_hidden, num_classes, seq_length)
    net = Network2(batch_size, rnn_hidden, seq_length)
    print(net)
    if GPU:
        net = net.cuda()

    # create inputs and targets
    inputs = torch.randn(seq_length, batch_size, 3, *input_size)
    targets = torch.ones(batch_size).type(torch.LongTensor)
    print('inputs:', inputs.size())
    print('targets:', targets.size())
    # store in Variables
    if GPU:
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
    else:
        inputs = Variable(inputs)
        targets = Variable(targets)

    # pass through network
    output = net.forward(inputs)
    print('output:', output.size())
    print(output)

    # compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, targets)
    print('loss:', loss)

    # clear existing gradients
    net.zero_grad()

    # back propagate
    loss.backward()

    # update weights
    params = list(net.lstm.parameters()) + list(net.linear.parameters())
    optimizer = torch.optim.SGD(params, lr=0.01)
    optimizer.step()


if __name__ == '__main__':
    main()


