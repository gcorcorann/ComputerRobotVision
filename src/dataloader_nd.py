#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import numbers
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms

class FlowDataset(Dataset):
    """Flow Attention Dataset.
    
    Args:
        labels_path (string):   path to text file with annotations
        transform (callable):   transform to be applied on image

    Returns:
        torch.utils.data.Dataset: dataset object
    """

    def __init__(self, labels_path, transform=None):
        # read video path and labels
        with open(labels_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = np.array(data)
            data = np.reshape(data, (-1, 2))

        np.random.shuffle(data)
        self.data = data
        self.transform = transform

    def __len__(self):
        """
        Retrieve Dataset Length.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve Next Item in Dataset.

        @return sample: sample['X'] contains input data while sample['y']
                        contains attention label.
        """
        video_path = self.data[idx, 0]
        # change to flow dataset
        video_path_flow = video_path[:45] + '_flow' + video_path[45:]
        y = int(self.data[idx, 1]) - 1
        X_flow = np.load(video_path_flow)
        # transform data
        if self.transform:
            X_flow = self.transform(X_flow)

        # reformat [numSeqs x numChannels x Height x Width]
        X_flow = np.transpose(X_flow, (0,3,1,2))
        # store in sample
        sample = {'X': X_flow, 'y': y}
        return sample

class TwostreamDataset(Dataset):
    """Twostream Attention Dataset.
    
    Args:
        labels_path (string):   path to text file with annotations
        transform (callable):   transform to be applied on image

    Returns:
        torch.utils.data.Dataset: dataset for raw image frames and ``flow
                                    images.``
    """

    def __init__(self, labels_path, transform=None):
        # read video path and labels
        with open(labels_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = np.array(data)
            data = np.reshape(data, (-1, 2))

        np.random.shuffle(data)
        self.data = data
        self.transform = transform

    def __len__(self):
        """
        Retrieve Dataset Length.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve Next Item in Dataset.

        @return sample: sample['X'] contains input data while sample['y']
                        contains attention label.
        """
        video_path = self.data[idx, 0]
        # change to flow dataset
        video_path_flow = video_path[:45] + '_flow' + video_path[45:]
        y = int(self.data[idx, 1]) - 1
        X = np.load(video_path)
        X_flow = np.load(video_path_flow)
        # transform data
        if self.transform:
            X = self.transform(X)
            X_flow = self.transform(X_flow)

        # combine into one stream (remove first sequence of appearance feats)
        X = X[1:]
        X_combine = np.vstack((X, X_flow))
        # reformat [numSeqs x numChannels x Height x Width]
        X = np.transpose(X_combine, (0,3,1,2))
        # store in sample
        sample = {'X': X, 'y': y}
        return sample

class AttentionDataset(Dataset):
    """Attention Level Dataset.
    
    Args:
        labels_path (string):   path to text file with annotations
        transform (callable):   transform to be applied on image

    Returns:
        torch.utils.data.Dataset: dataset object
    """

    def __init__(self, labels_path, transform=None):
        # read video paths and labels
        with open(labels_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = np.array(data)
            data = np.reshape(data, (-1, 2))
        
        np.random.shuffle(data)
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data[idx, 0]
        y = int(self.data[idx, 1]) - 1
        X = np.load(video_path)
        # transform data
        if self.transform:
            X_sample = self.transform(X)

        # reformat [numSeqs x numChannels x Height x Width]
        X_sample = np.transpose(X_sample, (0,3,1,2))
        # store in sample
        sample = {'X': X_sample, 'y': y}
        return sample

class Resize():
    """Resize frames in video sequence to a given size.

    Args:
        output_size (tuple): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be resized.

        Returns:
            ndarray: Resized video.
        """
        video_new = np.zeros(
                (len(video), *self.output_size, video.shape[3]),
                dtype=np.uint8
                )
        # resize each frame
        for idx, frame in enumerate(video):
            video_new[idx] = cv2.resize(frame, self.output_size)
            
        return video_new

class CenterCrop():
    """Crop frames in video sequence at the center.

    Args:
        output_size (tuple): Desired output size of crop.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be center-cropped.
        
        Returns:
            ndarray: Center-cropped video.
        """
        # hold transformed video
        video_new = np.zeros(
                (video.shape[0], *self.output_size, video.shape[3]),
                dtype=video.dtype
                )
        h, w = video.shape[1:3]
        new_h, new_w = self.output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        # center crop each frame
        for idx, frame in enumerate(video):
            video_new[idx] = frame[top: top + new_h, left: left + new_w]

        return video_new

class RandomCrop():
    """Crop randomly the frames in a video sequence.

    Args:
        output_size (tuple): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be cropped.

        Returns:
            ndarray: Cropped video.
        """
        # hold transformed video
        video_new = np.zeros(
                (video.shape[0], *self.output_size, video.shape[3]),
                dtype=video.dtype
                )
        h, w = video.shape[1:3]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # randomly crop each frame
        for idx, frame in enumerate(video):
            video_new[idx] = frame[top: top + new_h, left: left + new_w]

        return video_new

class RandomHorizontalFlip():
    """Horizontally flip a video sequence.

    Args:
        p (float): Probability of image being flipped. Default value is 0.5.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be flipped.

        Returns:
            ndarray: Randomly flipped video.
        """
        # check to perform flip
        if random.random() < self.p:
            # hold transformed video
            video_new = np.zeros_like(video)
            # flip each frame
            for idx, frame in enumerate(video):
                video_new[idx] = cv2.flip(frame, 1)

            return video_new

        return video

class RandomRotation():
    """Rotate video sequence by an angle.

    Args:
        degrees (float or int): Range of degrees to select from.
    """
    
    def __init__(self, degrees):
        assert isinstance(degrees, numbers.Real)
        self.degrees = degrees

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be rotated.

        Returns:
            ndarray: Randomly rotated video.
        """
        # hold transformed video
        video_new = np.zeros_like(video)
        h, w = video.shape[1:3]
        # random rotation
        angle = np.random.uniform(-self.degrees, self.degrees)
        # create rotation matrix with center point at the center of frame
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
        # rotate each frame
        for idx, frame in enumerate(video):
            video_new[idx] = cv2.warpAffine(frame, M, (w,h))
        
        return video_new

class Normalize():
    """Normalize video with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this
    transform will normalize each channge of the input video.

    Args:
        mean (list): Sequence of means for each channel.
        std (list): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        assert isinstance(mean, list)
        assert isinstance(std, list)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        # reverse order since images are read with opencv (i.e. BGR)
        self.mean = np.flip(self.mean, 0)
        self.std = np.flip(self.std, 0)

    def __call__(self, video):
        """
        Args:
            video (ndarray): Video to be normalized

        Returns:
            ndarray: Normalized video.
        """
        video = video / 255
        video = (video - self.mean) / self.std
        video = np.asarray(video, dtype=np.float32)
        return video

def get_loaders(labels_path, datatype, batch_size, num_workers, gpu=True):
    """Return dictionary of torch.utils.data.DataLoader.

    Args:
        labels_path (string):   path to text file with annotations
        datatype (string):     dataset being used
        batch_size (int):       number of instances in batch
        num_workers (int):      number of subprocesses used for data loading
        flow (bool):            if using flow dataset
        gpu (bool):             presence of gpu (default is true)

    Returns:
        torch.utils.data.DataLoader:    dataloader for custom dataset
        dictionary:                    dataset length for training and 
                                            validation
    """
    # data augmentation and normalization for training
    # just normalization for validation
    data_transforms = {
            'Train': transforms.Compose([
                Resize((256,256)),
                RandomCrop((224,224)),
                RandomHorizontalFlip(),
                RandomRotation(15),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'Valid': transforms.Compose([
                Resize((256,256)),
                CenterCrop((224,224)),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }

    # create dataset object
    if datatype == 'flow':
        datasets = {x: FlowDataset(
            labels_path, data_transforms[x]
            ) for x in ['Train', 'Valid']}
    elif datatype == 'raw':
        datasets = {x: AttentionDataset(
            labels_path, data_transforms[x]
            ) for x in ['Train', 'Valid']}
    elif datatype == 'twostream':
        datasets = {x: TwostreamDataset(
            labels_path, data_transforms[x]
            ) for x in ['Train', 'Valid']}

    # random split for training and validation
    num_instances = len(datasets['Train'])
    indices = list(range(num_instances))
    split = math.floor(num_instances * 0.8)
    train_indices, valid_indices = indices[:split], indices[split:]
    samplers = {'Train': SubsetRandomSampler(train_indices),
                'Valid': SubsetRandomSampler(valid_indices)}
    
    # dataset sizes
    dataset_sizes = {'Train': len(train_indices),
                     'Valid': len(valid_indices)}

    # create dataloders
    dataloaders = {
            x: DataLoader(datasets[x],
                batch_size=batch_size, sampler=samplers[x],
                num_workers=num_workers, pin_memory=gpu)
            for x in ['Train', 'Valid']
            }
    return dataloaders, dataset_sizes

def main():
    """Main Function."""
    import time

    # hyperparameters
    labels_path = '/usr/local/faststorage/gcorc/accv/average_labels.txt'
    batch_size = 32
    num_workers = 2
    gpu = torch.cuda.is_available()

    dataset = TwostreamDataset(labels_path)
    for i in range(len(dataset)):
        sample = dataset[i]
        data = sample['X']
        print(data.shape)
        if i == 0:
            break

#    # dictionary of dataloaders
#    dataloaders, dataset_sizes = get_loaders(labels_path, 'twostream', 
#            batch_size, num_workers, gpu=True)
#    print('Dataset Sizes:')
#    print(dataset_sizes)
#    print()
#
#    train_batch = next(iter(dataloaders['Train']))
#    data, labels = train_batch['X'], train_batch['y']
#    print(data.size())

if __name__ == '__main__':
    main()
