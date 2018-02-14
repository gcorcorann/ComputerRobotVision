#!/usr/bin/env python3
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import model
from dataloader import get_loaders

def train_network(dataloaders, network, criterion, optimizer, num_epochs, GPU):
    # store network to GPU
    if GPU:
        network = network.cuda()

    # store best validation accuracy
    best_model_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # start timer
        start = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                network.train(True)  # set model to training model
            else:
                network.train(False)  # set model to evaluation mode

            running_loss = 0.0
            running_correct = 0
            dataset_size = 0

            # iterate over data
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data['X'], data['y']
                # reshape into [numSeqs, batchSize, numChannels, Height, Width]
                inputs = inputs.transpose(0,1)

                # wrap in Variable
                if GPU:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                outputs = network.forward(inputs)
                
                # loss + predicted
                _, pred = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backwards + optimize only if in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(1)
                running_correct += torch.sum(pred == labels.data)
                dataset_size += inputs.size(1)
                
            # find size of dataset (numBatches * batchSize)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_correct / dataset_size
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'Valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(network.state_dict())

        # print elapsed time
        time_elapsed = time.time() - start
        print('Epoch complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        print()

    # load the best model weights
    network.load_state_dict(best_model_wts)
    return network, best_acc

def main():
    """Main Function."""
    # hyper-parameters
    GPU = torch.cuda.is_available()
    labels_path = '/home/gary/datasets/accv/labels_gary.txt'
    seq_length = 19
    input_size = (224,224)
    num_epochs = 2
    batch_size = 50
    rnn_hidden = 128
    num_classes = 4
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()

    # create dataloaders object
    dataloaders = get_loaders(labels_path, input_size, batch_size,
            num_workers=4)
    print('Training Dataset Batches:', len(dataloaders['Train']))
    print('Validation Dataset Batches:', len(dataloaders['Valid']))

    # create network and optimizer
    net = model.Network2(batch_size, rnn_hidden, seq_length)
    params = list(net.lstm.parameters()) + list(net.linear.parameters())
#    params = net.parameters()
    optimizer = optim.Adam(params, learning_rate)
    print(net)

    # train network
    net, best_acc = train_network(dataloaders, net, criterion, optimizer, 
            num_epochs, GPU)
    print('Best Validation Accuracy:', best_acc)

if __name__ == '__main__':
    main()
