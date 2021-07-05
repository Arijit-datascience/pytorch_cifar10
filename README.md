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
![image](https://user-images.githubusercontent.com/65554220/124408900-e1d17200-dd64-11eb-9a1f-0d4fc491152b.png)

## Plots

1. Train & Test Loss, Train & Test Accuracy  
![image](https://user-images.githubusercontent.com/65554220/124408182-4be91780-dd63-11eb-9c6a-85d552590731.png)  

2. Misclassified Images  
![image](https://user-images.githubusercontent.com/65554220/124408305-8a7ed200-dd63-11eb-9791-29ebc99a2e7a.png)  

3. Gradcam Images  
![image](https://user-images.githubusercontent.com/65554220/124408315-95396700-dd63-11eb-8df7-2b5a801d687a.png)  




