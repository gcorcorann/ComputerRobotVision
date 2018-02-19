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
    """
    Flow Attention Dataset.
    """
    def __init__(self, labels_path, transform=None):
        """
        @param  labels_path:    path to text file with annotations
                                    @pre: string
        @param  transform:      optional transform to be applied on image
                                    @pre: callable
        """
        # read video path and labels
        with open(labels_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = np.array(data)
            data = data.reshape(-1,2)

        np.random.shuffle(data)
        self.data = data
        self.transform = transform
        self.cap = cv2.VideoCapture()

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
        # list to store frames
        X = []
        y = int(self.data[idx, 1]) - 1
        video_path = self.data[idx, 0]
        # open video
        self.cap.open(video_path)
        # read initial frame
        _, frame1 = self.cap.read()
        # apply transform
        if self.transform:
            frame1 = self.transform(frame1)

        # convert to grayscale
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # for each remaining frame in video
        for i in range(99):
            # read second frame
            _, frame2 = self.cap.read()
            # sample video 5 frames apart
            if i != 0 and i % 5 == 0:
                # apply transform
                if self.transform:
                    frame2 = self.transform(frame2)

                # convert to grayscale
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                # compute flow
                flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None,
                        0.5, 3, 15, 3, 5, 1.2, 0)
                # normalize between 0-1
                cv2.normalize(flow, flow, 0, 1, cv2.NORM_MINMAX)
                # store flow in list
                X.append(flow)
                # set initial frame to second frame
                frame1 = frame2

        # store in ndarray
        X = np.array(X, dtype=np.float32)
        # reformat to [numSeqs x numChannels x Height x Width]
        X = np.transpose(X, (0,3,1,2))
        # store in sample
        sample = {'X': X, 'y': y}
        # release video capture device
        self.cap.release()
        return sample

class AttentionDataset(Dataset):
    """
    Attention Level Dataset.
    """
    def __init__(self, labels_path, transform=None):
        """
        @param  labels_path:    path to text file with annotations
                                    @pre: string
        @param  transform:     optional transform to be applied on image
                                    @pre: callable
        """
        # read video paths and labels
        with open(labels_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = np.array(data)
            data = np.reshape(data, (-1, 2))
        
        np.random.shuffle(data)
        self.data = data
        self.transform = transform
        self.cap = cv2.VideoCapture()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # list to store frames
        X = []
        y = int(self.data[idx, 1]) - 1
        video_path = self.data[idx, 0]
        # open video
        self.cap.open(video_path)
        for i in range(100):
            ret, frame = self.cap.read()
            if not ret:
                break
            # sample video 5 frames apart
            if i % 10 == 0:
                # store frame
                X.append(frame)

        # store in ndarray
        X = np.array(X, dtype=np.float32)
        # transform data
        if self.transform:
            X = self.transform(X)

        # store in sample
        sample = {'X': X, 'y': y}
        # release video capture device
        self.cap.release()
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
        # hold transformed video
        video_new = np.zeros(
                (video.shape[0], *self.output_size, video.shape[3]),
                dtype=video.dtype
                )
        # resize each frame
        for idx, frame in enumerate(video):
            frame = cv2.resize(frame, self.output_size)
            video_new[idx] = frame

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
            frame = frame[top: top + new_h, left: left + new_w]
            video_new[idx] = frame

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
            frame = frame[top: top + new_h, left: left + new_w]
            video_new[idx] = frame

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
                frame = cv2.flip(frame, 1)
                video_new[idx] = frame

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
            frame = cv2.warpAffine(frame, M, (w,h))
            video_new[idx] = frame
        
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
        video = np.transpose(video, (0,3,1,2))
        return video

def get_loaders(labels_path, batch_size, num_workers, gpu):
    """Return dictionary of torch.utils.data.DataLoader.

    Args:
        labels_path (string): path to text file with annotations
        batch_size (int): number of instances in batch
        num_workers (int): number of subprocesses used for data loading
        gpu (bool): presence of gpu

    Returns:
        torch.utils.data.DataLoader: dataloader for custom dataset
        dictionarny: dataset length for training and validation
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
    datasets = {x: AttentionDataset(labels_path, data_transforms[x])
            for x in ['Train', 'Valid']}

    # random split for training and validation
    num_instances = len(datasets['Train'])
    indices = list(range(num_instances))
    split = math.floor(num_instances * 0.8)
    train_indices, valid_indices = indices[:split][:10], indices[split:][:10]
    samplers = {'Train': SubsetRandomSampler(indices[:split]),
                'Valid': SubsetRandomSampler(indices[split:])}
    
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
    labels_path = '/home/gary/datasets/accv/labels_med.txt'
    batch_size = 5
    num_workers = 8
    gpu = torch.cuda.is_available()

    # dictionary of dataloaders
    dataloaders, dataset_sizes = get_loaders(labels_path, batch_size, 
            num_workers, gpu)
    print('Dataset Sizes:')
    print(dataset_sizes)
    print()

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1,2,0))
        # convert from BGR to RGB
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)

    # retrieve a batch of training data
    train_batch = next(iter(dataloaders['Valid']))
    data, labels = train_batch['X'], train_batch['y']

    classes = ['Low Attention', 'Medium Attention', 
            'High Attention', 'Very High Attention']
    print(classes)

    plt.figure()
    for i in range(batch_size):
        out = torchvision.utils.make_grid(data[i], nrow=10)
        plt.subplot(batch_size, 1, i+1), imshow(out, classes[labels[i]])
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
