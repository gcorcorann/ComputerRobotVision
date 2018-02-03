import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_loaders(batch_size, num_workers):
    """
    Get DataLoaders.

    @return torch.utils.data.DataLoader for CIFAR10
    """
    # image transforms
    transform = transforms.Compose([transforms.ToTensor(),
             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    
    # load datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
            download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR10(root='./data', train=False,
            download=True, transform=transforms.ToTensor())
    
    # dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)
    validloader = DataLoader(validset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)
    
    dataloaders = {'Train': trainloader, 'Valid': validloader}
    return dataloaders

def main():
    """Main Function."""
    batch_size = 10
    num_workers = 4
    dataloaders = get_loaders(batch_size, num_workers)
    # display images
    images, labels = next(iter(dataloaders['Train']))
    disp_images = torchvision.utils.make_grid(images, nrow=batch_size//2)
    disp_images = disp_images.numpy().transpose(1,2,0)
    plt.imshow(disp_images)
    plt.show()


if __name__ == '__main__':
    main()

