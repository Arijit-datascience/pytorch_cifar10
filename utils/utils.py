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
                border_mode=cv2.BORDER_CONSTANT,
                value=(norm_mean[0]*255, norm_mean[1]*255, norm_mean[2]*255)
            ),
            A.RandomCrop(
                height=32,
                width=32
            )
        ], p=0.5),
        A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=(norm_mean[0]*255, norm_mean[1]*255, norm_mean[2]*255), p=1),
        A.Rotate(limit=5),
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

def get_transforms_custom_resnet(norm_mean,norm_std):
    """get the train and test transform"""
    print(norm_mean,norm_std)
    train_transform = A.Compose(
        [
        A.Sequential([
            A.PadIfNeeded(
                min_height=40,
                min_width=40,
                border_mode=cv2.BORDER_CONSTANT,
                value=(norm_mean[0]*255, norm_mean[1]*255, norm_mean[2]*255)
            ),
            A.RandomCrop(
                height=32,
                width=32
            )
        ], p=1),
        A.HorizontalFlip(p=1),
        A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=(norm_mean[0]*255, norm_mean[1]*255, norm_mean[2]*255), p=1),
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
    dataloader_args = dict(shuffle=True,batch_size=512,num_workers=2, pin_memory=True) if cuda else dict(shuffle=True,batch_size=64,num_workers=1)

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)

    test_loader  = torch.utils.data.DataLoader(test_set, **dataloader_args)
    return(train_loader,test_loader)

def get_dataloaders_onecycle(train_set,test_set):

    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64, num_workers=1)

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
    fig, ax = plt.subplots(2,2, figsize=(25,15))
    
    ax[0,0].plot(np.array(train_loss), 'red', label="Training Loss")

    # Label the plot.
    ax[0,0].set_title("Training Loss")
    ax[0,0].set_xlabel("Epoch")
    ax[0,0].set_ylabel("Loss")
    ax[0,0].set_ylim(0, 2)

    ax[0,1].plot(np.array(test_loss), 'blue', label="Test Loss")

    # Label the plot.
    ax[0,1].set_title("Test Loss")
    ax[0,1].set_xlabel("Epoch")
    ax[0,1].set_ylabel("Loss")
    ax[0,1].set_ylim(0, 0.015)

    ax[1,0].plot(np.array(train_acc), 'red', label="Training Accuracy")

    # Label the plot.
    ax[1,0].set_title("Training Accuracy")
    ax[1,0].set_xlabel("Epoch")
    ax[1,0].set_ylabel("Loss")
    ax[1,0].set_ylim(20,92)

    ax[1,1].plot(np.array(test_acc), 'blue', label="Test Accuracy")

    # Label the plot.
    ax[1,1].set_title("Test Accuracy")
    ax[1,1].set_xlabel("Epoch")
    ax[1,1].set_ylabel("Loss")
    ax[1,1].set_ylim(30,92)

    plt.show()

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def wrong_predictions(model,test_loader, norm_mean, norm_std, classes, device):
    wrong_images=[]
    wrong_label=[]
    correct_label=[]
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])  

            wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))
        print(f'Total wrong predictions are {len(wrong_predictions)}')

        plot_misclassified(wrong_predictions, norm_mean, norm_std, classes)

    return wrong_predictions
    
def plot_misclassified(wrong_predictions, norm_mean, norm_std, classes):
    fig = plt.figure(figsize=(10,12))
    fig.tight_layout()
    for i, (img, pred, correct) in enumerate(wrong_predictions[:20]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j]*norm_std[j])+norm_mean[j]

        img = np.transpose(img, (1, 2, 0)) #/ 2 + 0.5
        ax = fig.add_subplot(5, 5, i+1)
        ax.axis('off')
        ax.set_title(f'\nactual : {classes[target.item()]}\npredicted : {classes[pred.item()]}',fontsize=10)
        ax.imshow(img)

    plt.show()
    
