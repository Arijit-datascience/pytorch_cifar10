# CIFAR10 image recognition using ResNet-18 architecture

## Code Structure

* [Model file](/model/resnet.py): This describes the ResNet-18 architecture with Layer Normalization  
<i>Referrence: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py</i>  

* [utils](/utils/utils.py): Utils code contains the below things:-  
  1. Data Loaders  
  2. Albumentations  
  3. Accuracy Plots
  4. Misclassification Image Plots
  5. Seed

* [main file](/main.py): Main code contains the below things:-  
  1. Train code
  2. Test code
  3. Main function for training and testing the model  

* [Colab file](/pytorch_cifar10_resnet.ipynb): The Google Colab file contains the below things:-  
1. Cloning the GIT Repository
2. Loading data calling the data loader function from utils file
3. Model Summary
4. Running the model calling the main file
5. Plotting Accuracy Plots
6. Plotting 20 Misclassification Images
7. Plotting the Gradcam for same 20 misclassified images

## Model Summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
         GroupNorm-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
         GroupNorm-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
         GroupNorm-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
         GroupNorm-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
        GroupNorm-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 32, 32]          73,728
        GroupNorm-14          [-1, 128, 32, 32]             256
           Conv2d-15          [-1, 128, 32, 32]         147,456
        GroupNorm-16          [-1, 128, 32, 32]             256
           Conv2d-17          [-1, 128, 32, 32]           8,192
        GroupNorm-18          [-1, 128, 32, 32]             256
       BasicBlock-19          [-1, 128, 32, 32]               0
           Conv2d-20          [-1, 128, 32, 32]         147,456
        GroupNorm-21          [-1, 128, 32, 32]             256
           Conv2d-22          [-1, 128, 32, 32]         147,456
        GroupNorm-23          [-1, 128, 32, 32]             256
       BasicBlock-24          [-1, 128, 32, 32]               0
           Conv2d-25          [-1, 256, 16, 16]         294,912
        GroupNorm-26          [-1, 256, 16, 16]             512
           Conv2d-27          [-1, 256, 16, 16]         589,824
        GroupNorm-28          [-1, 256, 16, 16]             512
           Conv2d-29          [-1, 256, 16, 16]          32,768
        GroupNorm-30          [-1, 256, 16, 16]             512
       BasicBlock-31          [-1, 256, 16, 16]               0
           Conv2d-32          [-1, 256, 16, 16]         589,824
        GroupNorm-33          [-1, 256, 16, 16]             512
           Conv2d-34          [-1, 256, 16, 16]         589,824
        GroupNorm-35          [-1, 256, 16, 16]             512
       BasicBlock-36          [-1, 256, 16, 16]               0
           Conv2d-37            [-1, 512, 8, 8]       1,179,648
        GroupNorm-38            [-1, 512, 8, 8]           1,024
           Conv2d-39            [-1, 512, 8, 8]       2,359,296
        GroupNorm-40            [-1, 512, 8, 8]           1,024
           Conv2d-41            [-1, 512, 8, 8]         131,072
        GroupNorm-42            [-1, 512, 8, 8]           1,024
       BasicBlock-43            [-1, 512, 8, 8]               0
           Conv2d-44            [-1, 512, 8, 8]       2,359,296
        GroupNorm-45            [-1, 512, 8, 8]           1,024
           Conv2d-46            [-1, 512, 8, 8]       2,359,296
        GroupNorm-47            [-1, 512, 8, 8]           1,024
       BasicBlock-48            [-1, 512, 8, 8]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 27.00
Params size (MB): 42.63
Estimated Total Size (MB): 69.64
----------------------------------------------------------------

## Plots

1. Train & Test Loss, Train & Test Accuracy  
![image](https://user-images.githubusercontent.com/65554220/124408182-4be91780-dd63-11eb-9c6a-85d552590731.png)  

2. Misclassified Images  
![image](https://user-images.githubusercontent.com/65554220/124408305-8a7ed200-dd63-11eb-9791-29ebc99a2e7a.png)  

3. Gradcam Images  
![image](https://user-images.githubusercontent.com/65554220/124408315-95396700-dd63-11eb-8df7-2b5a801d687a.png)  




