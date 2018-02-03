#!/usr/bin/env python3
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import LeNet5
from cifar_loader import get_loaders

def train_network(dataloaders, network, criterion, optimizer, num_epochs, GPU):
    # store network to GPU
    if GPU:
        network = network.cuda()

    # store best validation accuracy
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
                inputs, labels = data

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
                # TODO inputs.size(1)
                running_loss += loss.data[0] * inputs.size(0)
                running_correct += torch.sum(pred == labels.data)
                dataset_size += inputs.size(0)
                
            # find size of dataset (numBatches * batchSize)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_correct / dataset_size
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'Valid' and epoch_acc > best_acc:
                best_acc = epoch_acc

        # print elapsed time
        time_elapsed = time.time() - start
        print('Epoch complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        print()

    return best_acc

def main():
    """Main Function."""
    # hyper-parameters
    GPU = torch.cuda.is_available()
#    labels_path = '/home/gary/datasets/accv/labels_gary.txt'
#    seq_length = 19
#    input_size = (224,224,2)
    num_epochs = 20
    batch_size = 10
#    rnn_hidden = 128
#    num_classes = 4
    learning_rate = 1e-3
    criterion = nn.CrossEntropyLoss()

    # create dataloaders object
    dataloaders = get_loaders(batch_size, num_workers=4)
    print('Training Dataset:', len(dataloaders['Train'].dataset))
    print('Validation Dataset:', len(dataloaders['Valid'].dataset))

    # create network and optimizer
    net = LeNet5()
    print(net)
    optimizer = optim.Adam(net.parameters(), learning_rate)

    # train network
    best_acc = train_network(dataloaders, net, criterion, optimizer, 
            num_epochs, GPU)
#    print('Best Validation Accuracy:', best_acc)

if __name__ == '__main__':
    main()
