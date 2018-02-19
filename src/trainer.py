#!/usr/bin/env python3
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import model
from dataloader import get_loaders

def train_network(network, dataloaders, dataset_sizes, criterion, 
        optimizer, num_epochs, GPU):
    """Train network.

    Args:
        network (torchvision.models): network to train
        dataloaders (dictionary): contains torch.utils.data.DataLoader for both
            training and validation
        dataset_sizes (dictionary): size of training and validation datasets
        criterion (torch.nn.modules.loss): loss function
        opitimier (torch.optim): optimization algorithm.
        num_epochs (int): number of epochs used for training
        GPU (bool): gpu availability

    Returns:
        torchvision.models: best trained model
        float: best validaion accuracy
    """
    # store network to GPU
    if GPU:
        network = network.cuda()

    # store best validation accuracy
    best_model_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # start timer
        start = time.time()
        print()
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
                # reshape [numSeqs, batchSize, numChannels, Height, Width]
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
                correct = torch.sum(pred == labels.data)

                # backwards + optimize only if in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(1)
                running_correct += correct
                
            # find size of dataset (numBatches * batchSize)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'Valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(network.state_dict())

    # print elapsed time
    time_elapsed = time.time() - start
    print('Training Complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

    # load the best model weights
    network.load_state_dict(best_model_wts)
    # save the best model weights to disk
    torch.save(network.state_dict(), '../data/net_params.pkl')
    return network, best_acc

def main():
    """Main Function."""
    # dataloader parameters
    GPU = torch.cuda.is_available()
    labels_path = '/home/gary/datasets/accv/labels_med.txt'
    batch_size = 10
    # network parameters
    seq_length = 20
    input_size = (224,224)
    rnn_hidden = 128
    rnn_layers = 1
    # training parameters
    num_epochs = 5
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()

    # create dataloaders object
    dataloaders, dataset_sizes = get_loaders(labels_path, batch_size, 
            num_workers=8, gpu=GPU)
    print('Training Dataset:', dataset_sizes['Train'])
    print('Validation Dataset:', dataset_sizes['Valid'])

    # create network and optimizer
    net = model.VGG(rnn_hidden, rnn_layers)
    print(net)
    params = list(net.lstm.parameters()) + list(net.fc.parameters())
    optimizer = optim.Adam(params, learning_rate)

    # train network
    net, best_acc = train_network(net, dataloaders, dataset_sizes, 
            criterion, optimizer, num_epochs, GPU)
    print('Best Validation Accuracy:', best_acc)

if __name__ == '__main__':
    main()
