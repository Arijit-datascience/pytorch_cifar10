# Custom Resnet Model with prep layer, residual blocks with convolutions, 4x4 Max pooling and FC layer
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the ResNet class with all the conv layers and residual blocks for the cutom Resnet model
class ResNet(nn.Module):
    # Default of 10 classes for CIFAR-10 dataset
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        #Prep Layer with 3x64 kernels of 3x3, Batch Normalization & ReLU Activation
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        #Layer1: Consists of a convolutional layer and residual block
        
        ##Layer1-Part A: Convolutional layer with 64x128 kernels of size 3x3, MaxPooling 2x2, BN and ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()   
        )
        
        ##Layer1-Part B: Residual Block of 2 Convolutional layers with 128x128 kernels of size 3x3, BN and ReLU
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        #Layer2: Convolutional layer with 128x256 kernels of size 3x3, MaxPooling, BN and ReLU
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        #Layer3: This layer is similar to Layer1, except for the number of kernels: Consists of a convolutional layer and residual block
        
        ##Layer3-Part A: Convolutional layer with 256x512 kernels of size 3x3, MaxPooling 2x2, BN and ReLU
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        ##Layer3-Part B: Residual Block of 2 Convolutional layers with 512x512 kernels of size 3x3, BN and ReLU
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        ## MaxPooling of 4x4 to reduce the channel size to 1x1
        self.max = nn.Sequential(
            nn.MaxPool2d(4, 4)
        )
        
        # Fully Connected Layer with 512 input connections and num_classes as the output
        self.linear = nn.Linear(512*1, num_classes)

    # Define the forward function to build the model
    def forward(self, x):
        
        # Apply the Prep Convolution to the input
        out = self.prep(x)
        
        # The output of Prep Convolutional Layer is passed to the Layer-1
        
        # Layer1-PartA: Convolution
        X = self.conv2(out)
        # Layer1-PartB: Residual Block
        R1 = self.conv3(X)
        R1 = self.conv4(R1)
        
        # Residual-1: Add the input and output of the Residual Block-1
        out = X + R1
        
        #Layer2: Convolution on top of Residual-1 output
        out = self.conv5(out)

        # Layer3
        # Layer3-PartA: Convolution
        X2 = self.conv6(out)
        # Layer3-PartB: Residual Block-2
        R2 = self.conv7(X2)
        R2 = self.conv8(R2)
        
        # Residual-2: Add the input and output of the Residual Block-2
        out = X2 + R2
        
        # Maxpooling of 4x4 to reduce channel size
        out = self.max(out)
        
        # Flatten the output from MaxPool layer
        out = out.view(out.size(0), -1)
        
        # FC layer to return the num_classes as output
        out = self.linear(out)

        return out
