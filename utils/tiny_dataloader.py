import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import os
import numpy as np
import cv2
import time

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

# Giving each folder a ID
def get_id_dictionary(path):
    """This Function will genrate Id's for all classes
    Args:
        path (sting): file path to the classes txt file
    Returns:
        dict: key-class, value-Id
    """
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict

def get_data(path):
    """This function will create a list of file path to the image and their respective labels
    and will split them in training and testing samples
    Args:
        path (sting): file path to the classes txt file 
    Returns:
        list: training and testing images and labels 
    """
    id_dict = get_id_dictionary(path)
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()

    for key, value in id_dict.items():
        #train_data += [cv2.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), cv2.COLOR_BGR2RGB) for i in range(500)]
        train_data += [path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)) for i in range(500)]
        train_labels += [value for i in range(500)]
    
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)
    print('finished loading data, in {} seconds'.format(time.time() - t))
    print('Samples for training: {}'.format(len(X_train)))
    print('Samples for testing: {}'.format(len(X_test)))

    return X_train, X_test, y_train, y_test

class ImagenetDataset(Dataset):
    """Pytoch class to generate data loaders for Tiny Image Net Dataset
    Args:
        Dataset (pytorch class):
    """
    def __init__(self, path, labels, transforms=None):
        """
        Args:
            path (list): list containing path of images 
            labels (list): respective labels for images 
            transforms (albumentations compose class, optional): Contains Image transformations to be applied. Defaults to None
        """
        self.transform = transforms
        self.path, self.labels = path, labels

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        """genrate data and label
        Args:
            idx (int): index of sample
        Returns:
            tensor: tansformed image and label
        """
        label = self.labels[idx]
        image = cv2.imread(self.path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
          # Apply transformations
          image = self.transform(image=image)['image']
          #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return image, label

    def get_class_to_id_dict(self, path, id_dict):
        """Create a dict of label to class 
        Args:
            path (string): file path to the classes txt file 
            id_dict (dict):  key-class, value-Id
        Returns:
            dict: get class of respective label
        """
        all_classes = {}
        result = {}
        for i, line in enumerate(open( path + 'words.txt', 'r')):
            n_id, word = line.split('\t')[:2]
            all_classes[n_id] = word

        for key, value in id_dict.items():
            result[value] = (key, all_classes[key])      
        return result

def get_mean_std(loader):
    """Calculate mean and standard deviation of the dataset
    Args:
        loader (instance): torch instance for data loader
    Returns:
        tensor: mean and std of data
    """
    channel_sum, channel_squared_sum,  num_batches = 0,0,0
    
    for img,_ in loader:
        channel_sum += torch.mean(img/255., dim=[0,1,2])
        channel_squared_sum += torch.mean((img/255.)**2, dim=[0,1,2])
        num_batches += 1
        
    mean = channel_sum / num_batches
    std = (channel_squared_sum/num_batches - mean**2)**0.5
    print("The mean of dataset : ", mean)
    print("The std of dataset : ", std)
    return(tuple(map(lambda x: np.round(x,3), mean)), tuple(map(lambda x: np.round(x,3), std)))

def get_transforms(mean,std):
    train_transform = A.Compose([
      A.PadIfNeeded(min_height=76, min_width=76, always_apply=True),
      A.RandomCrop(64,64),
      A.Rotate(limit=15),
      A.CoarseDropout(1,24, 24, 1, 8, 8,fill_value=[m*255 for m in mean], mask_fill_value=None),
      A.VerticalFlip(),
      A.HorizontalFlip(),
      A.Normalize(mean, std),
      ToTensorV2()])

    test_transform = A.Compose([A.Normalize(mean, std),ToTensorV2()])
    
    return(train_transform,test_transform)

def get_dataloaders(X_train, X_test, y_train, y_test):
    
    X_train, X_test, y_train, y_test = train_test_data()
    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments
    dataloader_args = dict(shuffle=True,batch_size=512,num_workers=2, pin_memory=True) if cuda else dict(shuffle=True,batch_size=64,num_workers=1)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(ImagenetDataset(X_train, y_train, train_transform) , **dataloader_args)
    test_loader = torch.utils.data.DataLoader(ImagenetDataset(X_test, y_test, test_transform), **dataloader_args)
    return(train_loader,test_loader)
