#!/usr/bin/env python3
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

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
        self.cap.open(video_path)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # apply transform
            if self.transform:
                frame = self.transform(frame)

            # store frame
            X.append(frame)

        # store in ndarray
        X = np.array(X, dtype=np.float32)
        # reformat to [numSeqs x numChannels x Height x Width]
        X = np.transpose(X, (0,3,1,2))
        # store in sample
        sample = {'X': X, 'y': y}
        # release video capture device
        self.cap.release()
        return sample

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

class Resize():
    """
    Resizes the image to a given size.

    @param  output_size:    expected output image size
                                @pre: tuple

    @return image:          resized image
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, image):
        image = cv2.resize(image, self.output_size)
        return image

class Normalize():
    """
    Rescales image to [0-1], normalizes across channels with given mean and
    standard deviation and returns image in RGB formate.

    @param  mu:     channel means
                        @pre: list
    @param  std:    channel standard deviation
                        @pre: list

    @return image:  normalized image
    """
    def __init__(self, mu, std):
        assert isinstance(mu, list)
        assert isinstance(std, list)
        self.mu = mu
        self.std = std

    def __call__(self, image):
        image = image / 255
        b, g, r = cv2.split(image)
        # normalize image
        b = (b - self.mu[2]) / self.std[2]
        g = (g - self.mu[1]) / self.std[1]
        r = (r - self.mu[0]) / self.std[0]
        # combine into RGB format
        image = cv2.merge((r, g, b))
        return image

def get_loaders(labels_path, input_size, batch_size, num_workers):
    """
    Get DataLoaders.

    @param  labels_path:    path to text file with annotations
                                @pre: string
    @param  input_size:     resize frame to this size
                                @pre: tuple of integers

    @return torch.utils.data.DataLoader for custom dataset
    """
    # data transforms
#    composed = transforms.Compose([
#        Resize(input_size),
#        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ])
    # create dataset
#    dataset = AttentionDataset(labels_path, transform=composed)
    dataset = FlowDataset(labels_path, transform=Resize(input_size[:2])) 

    # split dataset into training and validation
    num_instances = len(dataset)
    indices = list(range(num_instances))
    split = int(np.floor(num_instances * 0.9))
    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # check if GPU is available
    gpu = torch.cuda.is_available()
    # dataloaders
    dataloaders = {
            'Train': DataLoader(dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=gpu),
            'Valid': DataLoader(dataset,
                batch_size=batch_size,
                sampler=valid_sampler,
                num_workers=num_workers,
                pin_memory=gpu)
            }

    return dataloaders

def main():
    """Main Function."""
    import time

    labels_path = '/home/gary/datasets/accv/labels_gary.txt'
    input_size = (224, 224)
    batch_size = 10

    # get DataLoaders
    dataloaders = get_loaders(labels_path, input_size, batch_size,
            num_workers=4)

    print('Training Size:', len(dataloaders['Train']) * batch_size)
    print('Validation Size:', len(dataloaders['Valid']) * batch_size)

    start = time.time()
    for i, sampled_batch in enumerate(dataloaders['Valid']):
        print(i, sampled_batch['X'].size(), sampled_batch['y'].size())
        if i == 3:
            break
    print('Elapsed Time:', time.time()-start)

if __name__ == '__main__':
    main()
