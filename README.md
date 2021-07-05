# CIFAR10 image recognition using ResNet-18 architecture

## Modelling Techniques

  1. Model: ResNet18
  2. Total Train data: 60,000 | Total Test Data: 10,000
  3. Total Parameters: 11,173,962
  4. Test Accuracy: 90.03%
  5. Epochs: Run till 40 epochs
  6. Normalization: Layer Normalization
  7. Regularization: L2 with factor 0.0001
  8. Optimizer: Adam with learning rate 0.001
  9. Loss criterion: Cross Entropy
  10. Scheduler: ReduceLROnPlateau
  11. Albumentations: 
      1. RandomCrop(32, padding=4)
      2. CutOut(16x16)
      3. Rotate(5 degree)
      4. CoarseDropout
      5. Normalization 
   12. Misclassified Images: 997 images were misclassified out of 10,000

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




