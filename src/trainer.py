#!/usr/bin/env python3
import time
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import model
from dataloader_nd import get_loaders

def train_network(network, dataloaders, dataset_sizes, criterion, 
        optimizer, scheduler, num_epochs):
    """Train network.

    Args:
        network (torchvision.models): network to train
        dataloaders (dictionary): contains torch.utils.data.DataLoader 
                                    for both training and validation
        dataset_sizes (dictionary): size of training and validation datasets
        criterion (torch.nn.modules.loss): loss function
        opitimier (torch.optim): optimization algorithm.
        scheduler (torch.optim): scheduled learning rate.
        num_epochs (int): number of epochs used for training

    Returns:
        torchvision.models: best trained model
        float: best validaion accuracy
        dictionary: training and validation losses
        dictionary: training and validation accuracy
    """
    # start timer
    start = time.time()
    # store network to GPU
    network = network.cuda()

    # store best validation accuracy
    best_model_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0
    losses = {'Train': [], 'Valid': []}
    accuracies = {'Train': [], 'Valid': []}

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        # each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                network.train(True)  # set model to training model
            else:
                network.train(False)  # set model to evaluation mode

            # used for accuracy and losses
            running_loss = 0.0
            running_correct = 0
            # iterate over data
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data['X'], data['y']
                # reshape [numSeqs, batchSize, numChannels, Height, Width]
                inputs = inputs.transpose(0,1)
                # wrap in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
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
            # store stats
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            # perform scheduled learning rate reduction
            if phase == 'Valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    # deep copy the model
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(network.state_dict())

    # print elapsed time
    time_elapsed = time.time() - start
    print()
    print('Training Complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
        ))

    # load the best model weights
    network.load_state_dict(best_model_wts)
    return network, best_acc, losses, accuracies

def plot_data(losses, accuracies, name):
    """Plot training and validation statistics.

    Args:
        losses (dictionary): containing list of cross entrophy losses for
                                training and validation splits
        accuracies (dictionary): contains list of accuracies for training
                                    and validation splits
        name (string): name to save plot
    """
    # set fontsize
    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
    ax1.set_title('Losses')
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.set_ylim(0,2)
    ax1.plot(losses['Train'], label='Training')
    ax1.plot(losses['Valid'], label='Validation')
    ax1.legend(loc='upper right')

    ax2.set_title('Accuracy')
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0,100)
    ax2.plot(accuracies['Train'], label='Training')
    ax2.plot(accuracies['Valid'], label='Validation')
    ax2.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig('../outputs/' + name)

def plot_confusion(network, dataloader, dataset_size):
    """Plot confusion matrix.

    Args:
        network (torchvision.models): network to evaluate
        dataloader (torch.utils.data.dataloader): validation dataloader
        dataset_size (int): size of validation dataset
    """
    import matplotlib.ticker as ticker

    # set fontsize
    plt.rcParams.update({'font.size': 12})
    # initialize confusion matrix
    confusion = torch.zeros(4,4)
    running_correct = 0
    for data in dataloader:
        # get inputs
        inputs, labels = data['X'], data['y']
        # wrap in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # reshape [numSeqs, batchSize, numChannels, Height, Width]
        inputs = inputs.transpose(0,1)
        # forward pass
        outputs = network.forward(inputs)
        # accuracy
        _, pred = torch.max(outputs.data, 1)
        for i in range(pred.size(0)):
            confusion[labels.data[i]][pred[i]] += 1
                
    # normalize confusion matrix
    for i in range(4):
        confusion[i] = confusion[i] / confusion[i].sum()

    print('Confusion Matrix:')
    print(confusion)
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(6,5))
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    all_categories = ['low attention', 'medium attention', 'high attention',
            'very high attention']
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.tight_layout()
    fig.savefig('../outputs/confusion.png')

def main():
    """Main Function."""
    # dataloader parameters
    if not torch.cuda.is_available():
        print('Please run with GPU')
        return

    labels_path = '/usr/local/faststorage/gcorc/accv/average_labels.txt'
    batch_size = 32
    num_workers = 2
    # network parameters
    sample_rate = 5
    rnn_hidden = 512
    rnn_layer = 2
    # training parameters
    num_epochs = 2
    learning_rate = 1e-4
    criterion = nn.CrossEntropyLoss()
    print('Training Parameters:')
    print('batch_size: {}, num_workers: {}, lr: {}'.format(
        batch_size, num_workers, learning_rate
        ))
    s = 'sample_rate: {}, rnn_layer: {}, rnn_hidden: {}'
    print(s.format(sample_rate, rnn_layer, rnn_hidden))
    # string to save plot
    s = 'bs:' + str(batch_size) + '_w:' + str(num_workers) + '_lr:' \
            + str(learning_rate) + '_rate:' + str(sample_rate) \
            + '_lay:' + str(rnn_layer) + '_hid:' + str(rnn_hidden) \
            + '.png'
    dataloaders, dataset_sizes = get_loaders(labels_path,
            batch_size, sample_rate, num_workers=num_workers, flow=False)
    print(dataset_sizes)
    # create network and optimizer
    net = model.VGGNetLSTMfc1(rnn_hidden, rnn_layer)
    #TODO change which parameters are trainable
#    params = net.parameters()
    params = list(net.lstm.parameters()) \
            + list(net.fc.parameters())
    optimizer = optim.Adam(params, learning_rate)
    # decay lr when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, 
            patience=500, verbose=True
            )
    # train the network
    net, val_acc, losses, accuracies = train_network(net, 
            dataloaders, dataset_sizes, criterion, optimizer, 
            scheduler, num_epochs)
    print('Best Validation Acc:', val_acc)
    print('-' * 60)
    print()
    # plot
    plot_data(losses, accuracies, s)
    plot_confusion(net, dataloaders['Valid'], dataset_sizes['Valid'])


#    best_params = {'sample_rate': sample_rate, 
#            'rnn_layer': rnn_layer, 'rnn_hidden': rnn_hidden}
#    torch.save(net.state_dict(), '../data/net_params.pkl')

if __name__ == '__main__':
    main()
