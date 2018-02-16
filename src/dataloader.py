#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import numbers
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
        # open video
        self.cap.open(video_path)
        for i in range(100):
            ret, frame = self.cap.read()
            if not ret:
                break
            # sample video 5 frames apart
            if i % 20 == 0:
                # store frame
                X.append(frame)

        # store in ndarray
        X = np.array(X, dtype=np.float32)
        # store in sample
        sample = {'X': X, 'y': y}
        # transform sample
        if self.transform:
            sample = self.transform(sample)

        # release video capture device
        self.cap.release()
        return sample

class Rescale():
    """
    Rescale the video in a sample to a given size.

    Args:
        output_size (tuple): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        video, label = sample['X'], sample['y']
        # hold transformed video
        video_new = np.zeros(
                (video.shape[0], *self.output_size, video.shape[3]),
                dtype=video.dtype
                )
        # resize each frame
        for idx, frame in enumerate(video):
            frame = cv2.resize(frame, self.output_size)
            video_new[idx] = frame
        return {'X': video_new, 'y': label}

class CenterCrop():
    """Crops given video sequence at the center.

    Args:
        output_size (tuple): Desired output size of crop.
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        """
        Args:
            sample (dictionary): sample['X'] contains video to be cropped.
        
        Returns:
            dictionary: sample['X'] contains cropped video.
        """
        video, label = sample['X'], sample['y']
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

        return {'X': video_new, 'y': label}
        
class RandomCrop():
    """Crop randomly the video in a sample.

    Args:
        output_size (tuple): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        video, label = sample['X'], sample['y']
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

        return {'X': video_new, 'y': label}

class RandomHorizontalFlip():
    """Horizontally flip the video in a sample.

    Args:
        p (float): Probability of image being flipped. Default value is 0.5.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            video (ndarray): Video to be flipped.

        Returns:
            ndarry: Randomly flipped video.
        """
        video, label = sample['X'], sample['y']
        # check to perform flip
        if random.random() < self.p:
            # hold transformed video
            video_new = np.zeros_like(video)
            # flip each frame
            for idx, frame in enumerate(video):
                frame = cv2.flip(frame, 1)
                video_new[idx] = frame

            return {'X': video_new, 'y': label}

        return {'X': video, 'y': label}

class RandomRotation():
    """Rotate video sequence by an angle.

    Args:
        degrees (float or int): Range of degrees to select from.
    """
    
    def __init__(self, degrees):
        assert isinstance(degrees, numbers.Real)
        self.degrees = degrees

    def __call__(self, sample):
        video, label = sample['X'], sample['y']
        # hold transformed video
        video_new = np.zeros_like(video)
        h, w = video.shape[1:3]
        # random rotation
        angle = np.random.uniform(-self.degrees, self.degrees)
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1)
        # rotate each frame
        for idx, frame in enumerate(video):
            frame = cv2.warpAffine(frame, M, (w,h))
            video_new[idx] = frame
        
        return {'X': video_new, 'y': label}


def main():
    """Main Function."""
    labels_path = '/home/gary/datasets/accv/labels_gary.txt'

    # create dataset object
    dataset = AttentionDataset(labels_path)

    # create transforms
    rescale = Rescale((256,256))
    randomcrop = RandomCrop((224,224))
    randomflip = RandomHorizontalFlip()
    centercrop = CenterCrop((224,224))
    randomrotate = RandomRotation(30)

    # extract single sample
    sample = dataset[65]

    # apply transforms
    sample_scale = rescale(sample)
    sample_crop = randomcrop(sample_scale)
    sample_flip = randomflip(sample_crop)
    sample_center = centercrop(sample_scale)
    sample_rotate = randomrotate(sample_flip)

    plt.figure(figsize=(25,20))
    for i in range(5):
        plt.subplot(5, 5, i+1)
        frame = sample_scale['X'][i]
        plt.imshow(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        plt.title('Scaled')

    for i in range(5):
        plt.subplot(5, 5, 5+i+1)
        frame = sample_crop['X'][i]
        plt.imshow(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        plt.title('Random Crop')

    for i in range(5):
        plt.subplot(5, 5, 10+i+1)
        frame = sample_flip['X'][i]
        plt.imshow(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        plt.title('Flip')

    for i in range(5):
        plt.subplot(5, 5, 15+i+1)
        frame = sample_center['X'][i]
        plt.imshow(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        plt.title('Center Crop')

    for i in range(5):
        plt.subplot(5, 5, 20+i+1)
        frame = sample_rotate['X'][i]
        plt.imshow(np.uint8(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        plt.title('Rotate')

    plt.show()

#class FlowDataset(Dataset):
#    """
#    Flow Attention Dataset.
#    """
#    def __init__(self, labels_path, transform=None):
#        """
#        @param  labels_path:    path to text file with annotations
#                                    @pre: string
#        @param  transform:      optional transform to be applied on image
#                                    @pre: callable
#        """
#        # read video path and labels
#        with open(labels_path, 'r') as f:
#            data = f.read()
#            data = data.split()
#            data = np.array(data)
#            data = data.reshape(-1,2)
#
#        np.random.shuffle(data)
#        self.data = data
#        self.transform = transform
#        self.cap = cv2.VideoCapture()
#
#    def __len__(self):
#        """
#        Retrieve Dataset Length.
#        """
#        return len(self.data)
#
#    def __getitem__(self, idx):
#        """
#        Retrieve Next Item in Dataset.
#
#        @return sample: sample['X'] contains input data while sample['y']
#                        contains attention label.
#        """
#        # list to store frames
#        X = []
#        y = int(self.data[idx, 1]) - 1
#        video_path = self.data[idx, 0]
#        # open video
#        self.cap.open(video_path)
#        # read initial frame
#        _, frame1 = self.cap.read()
#        # apply transform
#        if self.transform:
#            frame1 = self.transform(frame1)
#
#        # convert to grayscale
#        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#        # for each remaining frame in video
#        for i in range(99):
#            # read second frame
#            _, frame2 = self.cap.read()
#            # sample video 5 frames apart
#            if i != 0 and i % 5 == 0:
#                # apply transform
#                if self.transform:
#                    frame2 = self.transform(frame2)
#
#                # convert to grayscale
#                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#                # compute flow
#                flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None,
#                        0.5, 3, 15, 3, 5, 1.2, 0)
#                # normalize between 0-1
#                cv2.normalize(flow, flow, 0, 1, cv2.NORM_MINMAX)
#                # store flow in list
#                X.append(flow)
#                # set initial frame to second frame
#                frame1 = frame2
#
#        # store in ndarray
#        X = np.array(X, dtype=np.float32)
#        # reformat to [numSeqs x numChannels x Height x Width]
#        X = np.transpose(X, (0,3,1,2))
#        # store in sample
#        sample = {'X': X, 'y': y}
#        # release video capture device
#        self.cap.release()
#        return sample
#
#class Resize():
#    """
#    Resizes the image to a given size.
#
#    @param  output_size:    expected output image size
#                                @pre: tuple
#
#    @return image:          resized image
#    """
#    def __init__(self, output_size):
#        assert isinstance(output_size, tuple)
#        self.output_size = output_size
#
#    def __call__(self, image):
#        image = cv2.resize(image, self.output_size)
#        return image
#
#class Normalize():
#    """
#    Rescales image to [0-1], normalizes across channels with given mean and
#    standard deviation and returns image in RGB formate.
#
#    @param  mu:     channel means
#                        @pre: list
#    @param  std:    channel standard deviation
#                        @pre: list
#
#    @return image:  normalized image
#    """
#    def __init__(self, mu, std):
#        assert isinstance(mu, list)
#        assert isinstance(std, list)
#        self.mu = mu
#        self.std = std
#
#    def __call__(self, image):
#        image = image / 255
#        b, g, r = cv2.split(image)
#        # normalize image
#        b = (b - self.mu[2]) / self.std[2]
#        g = (g - self.mu[1]) / self.std[1]
#        r = (r - self.mu[0]) / self.std[0]
#        # combine into RGB format
#        image = cv2.merge((r, g, b))
#        return image
#
#def get_loaders(labels_path, input_size, batch_size, num_workers):
#    """
#    Get DataLoaders.
#
#    @param  labels_path:    path to text file with annotations
#                                @pre: string
#    @param  input_size:     resize frame to this size
#                                @pre: tuple of integers
#
#    @return torch.utils.data.DataLoader for custom dataset
#    """
#    # data transforms
#    composed = transforms.Compose([
#        Resize(input_size),
#        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ])
#    # create dataset
#    dataset = AttentionDataset(labels_path, transform=composed)
##    dataset = FlowDataset(labels_path, transform=Resize(input_size)) 
#
#    # split dataset into training and validation
#    num_instances = len(dataset)
#    indices = list(range(num_instances))
#    split = int(np.floor(num_instances * 0.8))
#    train_idx, valid_idx = indices[:split][:50], indices[split:][:1]
#    train_sampler = SubsetRandomSampler(train_idx)
#    valid_sampler = SubsetRandomSampler(valid_idx)
#
#    # check if GPU is available
#    gpu = torch.cuda.is_available()
#    # dataloaders
#    dataloaders = {
#            'Train': DataLoader(dataset,
#                batch_size=batch_size,
#                sampler=train_sampler,
#                num_workers=num_workers,
#                pin_memory=gpu),
#            'Valid': DataLoader(dataset,
#                batch_size=batch_size,
#                sampler=valid_sampler,
#                num_workers=num_workers,
#                pin_memory=gpu)
#            }
#
#    return dataloaders
#
#def main():
#    """Main Function."""
#    import time
#
#    # start timer
#    start = time.time()
#
#    labels_path = '/home/gary/datasets/accv/labels_gary.txt'
#    input_size = (224, 224)
#    batch_size = 10
#
#
#    dataset = AttentionDataset(labels_path)
#
#    for k in range(len(dataset)):
#        sample = dataset[k]
#        data, label = sample['X'], sample['y']
#        print(data.shape, label)
#
#        if k == 4:
#            break
#
##    # get DataLoaders
##    dataloaders = get_loaders(labels_path, input_size, batch_size,
##            num_workers=4)
##
##    print('Training Batches Size:', len(dataloaders['Train']))
##    print('Validation Batches Size:', len(dataloaders['Valid']))
##
##    # go through validation set
##    for i, sampled_batch in enumerate(dataloaders['Valid']):
##        print(i, sampled_batch['X'].size(), sampled_batch['y'].size())
##        if i == 3:
##            break
##
##    print('Elapsed Time: {} seconds'.format(time.time()-start))

if __name__ == '__main__':
    main()
