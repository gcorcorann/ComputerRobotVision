#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class SingleStream(nn.Module):
    """Single Stream (Spatial OR Temporal) + LSTM

    Args:
        model (string): stream CNN architecture
        rnn_hidden (int): number of hidden units in each rnn layer.
        rnn_layers (int): number of layers in rnn model.
    """

    def __init__(self, model, rnn_hidden, rnn_layers, pretrained=True,
            finetuned=True):
        super().__init__()
        if model is 'AlexNet':
            self.cnn = models.alexnet(pretrained)
            num_fts = self.cnn.classifier[4].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-3]
                    )
        elif model is 'VGGNet11':
            self.cnn = models.vgg11_bn(pretrained)
            num_fts = self.cnn.classifier[3].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-4]
                    )
        elif model is 'VGGNet16':
            self.cnn = models.vgg16_bn(pretrained)
            num_fts = self.cnn.classifier[3].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-4]
                    )
        elif model is 'VGGNet19':
            self.cnn = models.vgg19_bn(pretrained)
            num_fts = self.cnn.classifier[3].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-4]
                    )
        elif model is 'ResNet18':
            self.cnn = models.resnet18(pretrained)
            num_fts = self.cnn.fc.in_features
            self.cnn = nn.Sequential(
                    *list(self.cnn.children())[:-1]
                    )
        elif model is 'ResNet34':
            self.cnn = models.resnet34(pretrained)
            num_fts = self.cnn.fc.in_features
            self.cnn = nn.Sequential(
                    *list(self.cnn.children())[:-1]
                    )
        else:
            print('Please input correct model architecture')
            return

        for param in self.cnn.parameters():
            param.requires_grad = finetuned

        # add lstm layer
        self.lstm = nn.LSTM(num_fts, rnn_hidden, rnn_layers)
#        # add linear layer
#        self.fc = nn.Linear(rnn_hidden, 4)

    def forward(self, inputs):
        """Forward pass through network.

        Args:
            inputs (torch.Tensor): tensor of dimensions
                [numSeqs x batchSize x numChannels x Width x Height]

        Returns:
            torch.Tensor: final output of dimensions
                [batchSize x numClasses]
        """
        # list to hold features
        feats = []
        # for each input in sequence
        for inp in inputs:
            # pass through cnn
            outs = self.cnn.forward(inp).data
            outs = torch.squeeze(outs)
            feats.append(outs)
        
        # format features and store in Variable
        feats = torch.stack(feats)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
#        outputs = self.fc(outputs[-1])
        return outputs

class TwoStreamFusion(nn.Module):
    def __init__(self, rnn_hidden, rnn_layers):
        super().__init__()
        # spatial stream CNN
        self.spatial_stream = SingleStream('VGGNet19', rnn_hidden, rnn_layers)
        # temporal stream CNN
        self.temporal_stream = SingleStream('VGGNet16', rnn_hidden, rnn_layers)
        # fusion layer
        self.linear = nn.Linear(rnn_hidden*2, 4)

    def forward(self, inputs):
        # half sequence length
        seq_length = len(inputs)//2
        # split into appearance and flow images
        app_inputs = inputs[:seq_length]
        flow_inputs = inputs[seq_length:]
        # pass through spatial stream
        app_outs = self.spatial_stream.forward(app_inputs).data
        flow_outs = self.temporal_stream.forward(flow_inputs).data
        combined_outs = torch.cat((app_outs, flow_outs), 2)
        # fusion layer (only last outputs of each stream)
        out_fusion = self.linear(Variable(combined_outs[-1]))
#        out_fusion = nn.Tanh(out_fusion)
        return out_fusion


class TwoStream(nn.Module):
    """Two Stream (Spatial AND Temporal) + LSTM

    Args:
        rnn_hidden (int): number of hidden units in each rnn layer.
        rnn_layers (int): number of layers in rnn model.
    """

    def __init__(self, rnn_hidden, rnn_layers):
        super().__init__()
        # spatial stream
        self.spatial_cnn = models.vgg19_bn(pretrained=True)
        spat_num_fts = self.spatial_cnn.classifier[3].in_features
        self.spatial_cnn.classifier = nn.Sequential(
                *list(self.spatial_cnn.classifier.children())[:-4]
                )
        # temporal stream
        self.temporal_cnn = models.vgg16_bn(pretrained=True)
        temp_num_fts = self.temporal_cnn.classifier[3].in_features
        self.temporal_cnn.classifier = nn.Sequential(
                *list(self.temporal_cnn.classifier.children())[:-4]
                )
        # dropout layer after concatenation of both streams
        self.linear = nn.Linear(spat_num_fts + temp_num_fts, spat_num_fts)
        self.drop = nn.Dropout()
        # add lstm layer
        self.lstm = nn.LSTM(spat_num_fts, rnn_hidden, rnn_layers)
        # add linear layer
        self.fc = nn.Linear(rnn_hidden, 4)

    def forward(self, inputs):
        """Forward pass through network.

        Args:
            inputs (torch.Tensor): tensor of dimensions
                [numSeqs x batchSize x numChannels x Width x Height]

        Returns:
            torch.Tensor: final output of dimensions
                [batchSize x numClasses]
        """
        # list to hold features
        feats = []
        # half sequence length
        seq_length = len(inputs)//2
        # split into appearance and flow images
        app_inputs = inputs[:seq_length]
        flow_inputs = inputs[seq_length:]
        for i in range(seq_length):
            # pass through spatial stream
            app_outs = self.spatial_cnn.forward(app_inputs[i]).data
            flow_outs = self.temporal_cnn.forward(flow_inputs[i]).data
            combined_outs = torch.cat((app_outs, flow_outs), 1)
            # linear + dropout
            combined_outs = self.linear(Variable(combined_outs))
            combined_outs = self.drop(combined_outs).data
            feats.append(combined_outs)
        
        # format features and store in Variable
        feats = torch.stack(feats)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.fc(outputs[-1])
        return outputs

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
            out = F.relu(self.fc2(out))
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

class VGGNetLSTMfc2(nn.Module):
    """Pretrained VGG Net with LSTM.

    Args:
        rnn_hidden (int): number of hidden units in each rnn layer.
        rnn_layers (int): number of layers in rnn model.
    """

    def __init__(self, rnn_hidden, rnn_layers):
        super().__init__()
        self.cnn = models.vgg19_bn(pretrained=True)
        # number of input features of last layer of cnn
        num_ftrs = self.cnn.classifier[6].in_features
        # remove last layer
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
                )
        for param in self.cnn.parameters():
            param.requires_grad = False

        # add lstm layer
        self.lstm = nn.LSTM(num_ftrs, rnn_hidden, rnn_layers)
        # add linear layer
        self.fc = nn.Linear(rnn_hidden, 4)

    def forward(self, inputs):
        """Forward pass through network.

        Args:
            inputs (torch.Tensor): tensor of dimensions
                [numSeqs x batchSize x numChannels x Width x Height]

        Returns:
            torch.Tensor: final output of dimensions
                [batchSize x numClasses]
        """
        # list to hold features
        feats = []
        # for each input in sequence
        for inp in inputs:
            # pass through cnn
            outs = self.cnn.forward(inp)
            feats.append(outs)
        
        # format features and store in Variable
        feats = torch.stack(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.fc(outputs[-1])
        return outputs

class VGGNetLSTMfc1(nn.Module):
    """Pretrained VGG Net with LSTM.

    Args:
        rnn_hidden (int): number of hidden units in each rnn layer.
        rnn_layers (int): number of layers in rnn model.
    """

    def __init__(self, rnn_hidden, rnn_layers):
        super().__init__()
        self.cnn = models.vgg19_bn(pretrained=True)
        # number of inputs features in fc1
        num_ftrs = self.cnn.classifier[3].in_features
        # remove last two fc layers (need to remove ReLU + Dropout layers)
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-4]
                )
        #TODO change VGG parameters to trainable/un-trainable
        for param in self.cnn.parameters():
            param.requires_grad = False

        # add lstm layer
        self.lstm = nn.LSTM(num_ftrs, rnn_hidden, rnn_layers)
        # add linear layer
        self.fc = nn.Linear(rnn_hidden, 4)

    def forward(self, inputs):
        """Forward pass through network.

        Args:
            inputs (torch.Tensor): tensor of dimensions
                [numSeqs x batchSize x numChannels x Width x Height]

        Returns:
            torch.Tensor: final output of dimensions
                [batchSize x numClasses]
        """
        # list to hold features
        feats = []
        # for each input in sequence
        for inp in inputs:
            # pass through cnn
            outs = self.cnn.forward(inp).data
            print('outs:', outs.size())
            feats.append(outs)
        
        # format features and store in Variable
        feats = torch.stack(feats)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.fc(outputs[-1])
        return outputs

class VGGNetLSTMfc1Flow(nn.Module):
    """Pretrained VGG Net with LSTM.

    Args:
        rnn_hidden (int): number of hidden units in each rnn layer.
        rnn_layers (int): number of layers in rnn model.
    """

    def __init__(self, rnn_hidden, rnn_layers):
        super().__init__()
        self.cnn = models.vgg19_bn(pretrained=True)
        # number of inputs features in fc1
        num_ftrs = self.cnn.classifier[3].in_features
        # remove last two fc layers (need to remove ReLU + Dropout layers)
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-4]
                )
        #TODO change VGG parameters to trainable/un-trainable
        for param in self.cnn.parameters():
            param.requires_grad = False

        # add lstm layer
        self.lstm = nn.LSTM(num_ftrs, rnn_hidden, rnn_layers)
        # add linear layer
        self.fc = nn.Linear(rnn_hidden, 4)

    def forward(self, inputs):
        """Forward pass through network.

        Args:
            inputs (torch.Tensor): tensor of dimensions
                [numSeqs x batchSize x numChannels x Width x Height]

        Returns:
            torch.Tensor: final output of dimensions
                [batchSize x numClasses]
        """
        # list to hold features
        feats = []
        # for each input in sequence
        for inp in inputs:
            # pass through cnn
            outs = self.cnn.forward(inp).data
            feats.append(outs)
        
        # format features and store in Variable
        feats = torch.stack(feats)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.fc(outputs[-1])
        return outputs

class ResNetLSTM(nn.Module):
    """Pretrained ResNet with LSTM.

    Args:
        rnn_hidden (int): number of hidden units in each rnn layer.
        rnn_layers (int): number of layers in rnn model.
    """

    def __init__(self, rnn_hidden, rnn_layers):
        super().__init__()
        self.cnn = models.resnet152(pretrained=True)
        # number of input features of last layer of cnn
        num_ftrs = self.cnn.fc.in_features
        # remove last layer
        self.cnn = nn.Sequential(
                *list(self.cnn.children())[:-1]
                )
        for param in self.cnn.parameters():
            param.requires_grad = False

        # add lstm layer
        self.lstm = nn.LSTM(num_ftrs, rnn_hidden, rnn_layers)
        # add linear layer
        self.fc = nn.Linear(rnn_hidden, 4)

    def forward(self, inputs):
        """Forward pass through network.

        Args:
            inputs (torch.Tensor): tensor of dimensions
                [numSeqs x batchSize x numChannels x Width x Height]

        Returns:
            torch.Tensor: final output of dimensions
                [batchSize x numClasses]
        """
        # list to hold features
        feats = []
        # for each input in sequence
        for inp in inputs:
            # pass through cnn
            outs = self.cnn.forward(inp)
            # remove 1 dimensions
            outs = torch.squeeze(outs)
            feats.append(outs)
        
        # format features and store in Variable
        feats = torch.stack(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.fc(outputs[-1])
        return outputs

class ResNetLSTMFlow(nn.Module):
    """
    ResNet + LSTM.
    """
    def __init__(self, rnn_hidden, rnn_layers):
        """
        Initialize Parameters.
        
        rnn_hidden (int): number of hidden units in each rnn layer.
        rnn_layers (int): number of layers in rnn model.
        """
        super().__init__()
        # create ResNet model
        self.cnn = models.resnet18(pretrained=True)
        # number of input features of last layer of cnn
        num_ftrs = self.cnn.fc.in_features
        # remove last layer
        self.cnn = nn.Sequential(
                *list(self.cnn.children())[:-1]
                )
        # use CNN as feature extractor
#        for param in self.cnn.parameters():
#            param.requires_grad = False
        # lstm layer
        self.lstm = nn.LSTM(num_ftrs, rnn_hidden, rnn_layers)
        self.fc = nn.Linear(rnn_hidden, 4)
        
    def forward(self, inputs):
        # list to hold features
        feats = []
        # for each input in sequence
        for inp in inputs:
            # pass through CNN
            outs = self.cnn.forward(inp).data
            outs = torch.squeeze(outs)
            feats.append(outs)

        # format features and store in Variable
        feats = torch.stack(feats)
        feats = Variable(feats)
        # pass through LSTM
        outputs, _ = self.lstm(feats)
        outputs = self.fc(outputs[-1])
        return outputs

def main():
    import time

    # start timer
    start = time.time()
    
    # hyper-parameters
    GPU = torch.cuda.is_available()
    num_epochs = 200
    seq_length = 19
    batch_size = 10
    input_size = (224,224)
    rnn_hidden = 128
    rnn_layers = 1

    # create model object
    net = ResNetLSTMFlow(rnn_hidden, rnn_layers)
#    net = ResNetLSTM(rnn_hidden, rnn_layers)
    print(net)
    if GPU:
        net = net.cuda()

    # create inputs and targets
    inputs = torch.randn(seq_length, batch_size, 2, *input_size)
    targets = np.random.randint(4, size=batch_size)
    targets = torch.from_numpy(targets).type(torch.LongTensor)
    print('inputs:', inputs.size())
    print('targets:', targets.size())
    print('')
    # store in Variables
    if GPU:
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
    else:
        inputs = Variable(inputs)
        targets = Variable(targets)

    # training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # pass through network
        output = net.forward(inputs)
        print('output:', output.size())
    
        # compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, targets)
        print('loss:', loss.data[0])
        print('')
    
        # predicted
        _, pred = torch.max(output.data,1)
        correct = torch.sum(pred==targets.data)
        print('accuracy:', correct/batch_size)
        print('')

        # clear existing gradients
        net.zero_grad()
    
        # back propagate
        loss.backward()
    
        # update weights
#        params = list(net.lstm.parameters()) + list(net.fc.parameters())
        params = net.parameters()
        optimizer = torch.optim.SGD(params, lr=0.1)
        optimizer.step()

    time_elapsed = time.time() - start
    print('Elapsed Training Time: {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

#    # save and load only the model parameters
#    torch.save(net.state_dict(), 'params.pkl')
#    net.load_state_dict(torch.load('params.pkl'))

if __name__ == '__main__':
    main()


