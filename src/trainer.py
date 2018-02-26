#!/usr/bin/env python3
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import model
from dataloader_nd import get_loaders

def train_network(network, dataloaders, dataset_sizes, criterion, 
        optimizer, scheduler, num_epochs, GPU):
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
        GPU (bool): gpu availability

    Returns:
        torchvision.models: best trained model
        float: best validaion accuracy
    """
    # start timer
    start = time.time()
    # store network to GPU
    if GPU:
        network = network.cuda()

    # store best validation accuracy
    best_model_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0

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
    print('Training Complete in {:.0f}h {:.0f}m'.format(
        time_elapsed // (60*60), time_elapsed // 60 % 60
        ))

    # load the best model weights
    network.load_state_dict(best_model_wts)
    return network, best_acc

def main():
    """Main Function."""
    # dataloader parameters
    GPU = torch.cuda.is_available()
    labels_path = '/usr/local/faststorage/gcorc/accv/labels_med.txt'
    batch_size = 50
    num_workers = [0, 2, 4, 6]
    # network parameters
    sample_rates = [1, 2, 5, 10]
    rnn_hiddens = [32, 64, 128, 256, 512]
    rnn_layers = [1, 2, 3]
    # training parameters
    num_epochs = 30
    learning_rate = 1e-3
    criterion = nn.CrossEntropyLoss()

    # hold best validaion accuracy and parameters
    best_acc = 0
    best_params = {}
    # train for all sets of hyper parameters
    for i, sample_rate in enumerate(sample_rates):
        for rnn_layer in rnn_layers:
            for rnn_hidden in rnn_hiddens:
                print()
                print('Training Parameters:')
                s = 'sample_rate: {}, rnn_layer: {}, rnn_hidden: {}'
                print(s.format(sample_rate, rnn_layer, rnn_hidden))
                dataloaders, dataset_sizes = get_loaders(labels_path,
                        batch_size, sample_rate, num_workers=num_workers[i],
                        gpu=GPU)
                # create network and optimizer
                net = model.VGG(rnn_hidden, rnn_layer)
                params = list(net.lstm.parameters()) \
                        + list(net.fc.parameters())
                optimizer = optim.Adam(params, learning_rate)
                # decay lr when validation loss stops improving
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, 
                        patience=2, verbose=True
                        )
                # train the network
                net, val_acc = train_network(net, dataloaders, dataset_sizes, 
                        criterion, optimizer, scheduler, num_epochs, GPU)
                print('Best Validation Acc:', val_acc)
                print('-' * 70)
                if val_acc > best_acc:
                    best_params = {'sample_rate': sample_rate, 
                            'rnn_layer': rnn_layer, 'rnn_hidden': rnn_hidden}
                    torch.save(net.state_dict(), '../data/net_params.pkl')
                    best_acc = val_acc
                    
    print()
    print(best_params, 'Best Validation Acc:', best_acc)

if __name__ == '__main__':
    main()
