import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from torchsummary import summary

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

# Calculating Mean and Standard Deviation
def cifar10_mean_std():
    simple_transforms = transforms.Compose([
                                           transforms.ToTensor(),
                                           ])
    exp_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=simple_transforms)
    exp_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=simple_transforms)

    train_data = exp_train.data
    test_data = exp_test.data

    exp_data = np.concatenate((train_data,test_data),axis=0) # contatenate entire data
    exp_data = np.transpose(exp_data,(3,1,2,0)) # reshape to (60000, 32, 32, 3)

    norm_mean = (np.mean(exp_data[0])/255, np.mean(exp_data[1])/255, np.mean(exp_data[2])/255)
    norm_std   = (np.std(exp_data[0])/255, np.std(exp_data[1])/255, np.std(exp_data[2])/255)

    return(tuple(map(lambda x: np.round(x,3), norm_mean)), tuple(map(lambda x: np.round(x,3), norm_std)))

def get_transforms(norm_mean,norm_std):
    """get the train and test transform"""
    print(norm_mean,norm_std)
    train_transform = A.Compose(
        [
        A.Sequential([
            A.PadIfNeeded(
                min_height=40,
                min_width=40,
                border_mode=cv.BORDER_CONSTANT,
                value=norm_mean
            ),
            A.RandomCrop(
                height=32,
                width=32
            )
        ], p=1)
        A.Cutout (num_holes=1, max_h_size=16, max_w_size=16, fill_value=norm_mean, p=1)
        A.Normalize(norm_mean, norm_std),
        ToTensorV2()
    ]
    )

    test_transform = A.Compose(
        [
        A.Normalize(norm_mean, norm_std, always_apply=True),
        ToTensorV2()
    ]
    )
    
    return(train_transform,test_transform)

def get_datasets(train_transform,test_transform):
    
    class Cifar10_SearchDataset(datasets.CIFAR10):
        def __init__(self, root="./data", train=True, download=True, transform=None):
            super().__init__(root=root, train=train, download=download, transform=transform)
            
        def __getitem__(self, index):
            image, label = self.data[index], self.targets[index]

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]

            return image, label
    
    train_set = Cifar10_SearchDataset(root='./data', train=True,download=True, transform=train_transform)
    test_set  = Cifar10_SearchDataset(root='./data', train=False,download=True, transform=test_transform)

    return(train_set,test_set)

def get_dataloaders(train_set,test_set):

    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64, num_workers=1)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)

    test_loader  = torch.utils.data.DataLoader(test_set, **dataloader_args)
    return(train_loader,test_loader)

def show_sample_images(data_loader, classes, mean=.5, std=.5, num_of_images = 10, is_norm = True):
    """ Display images from a given batch of images """
    smpl = iter(data_loader)
    im,lb = next(smpl)
    plt.figure(figsize=(20,20))
    if num_of_images > im.size()[0]:
        num = im.size()[0]
        print(f'Can display max {im.size()[0]} images')
    else:
        num = num_of_images
        print(f'Displaying {num_of_images} images')
    for i in range(num):
        if is_norm:
            img = im[i].squeeze().permute(1,2,0)*std+mean
        plt.subplot(10,10,i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(classes[lb[i]],fontsize=15)

def valid_accuracy_loss_plots(train_loss, train_acc, test_loss, test_acc):

    # Use plot styling from seaborn.
    sns.set(style='whitegrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1)
    plt.rcParams["figure.figsize"] = (25,6)

    # Plot the learning curve.
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(np.array(train_loss), 'red', label="Training Loss")
    ax1.plot(np.array(test_loss), 'blue', label="Validation Loss")

    # Label the plot.
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0.3,1)
    ax1.legend()

    ax2.plot(np.array(train_acc), 'red', label="Training Accuracy")
    ax2.plot(np.array(test_acc), 'blue', label="Validation Accuracy")

    # Label the plot.
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_ylim(30,90)
    ax2.legend()

    plt.show()

def misclassification(predictions, targets, data):
    pred = predictions.view(-1)
    target = targets.view(-1)

    index = 0
    misclassified_image = []

    for label, predict in zip(target, pred):
        if label != predict:
            misclassified_image.append(index)
        index += 1

    plt.figure(figsize=(10,5))
    plt.suptitle('Misclassified Images');

    for plot_index, bad_index in enumerate(misclassified_image[0:10]):
        p = plt.subplot(2, 5, plot_index+1)
        img = data.squeeze().permute(1,2,0)
        p.imshow(img[bad_index].reshape(3,32,32))
        p.axis('off')
        p.set_title(f'Pred:{pred[bad_index]}, Actual:{target[bad_index]}')

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
