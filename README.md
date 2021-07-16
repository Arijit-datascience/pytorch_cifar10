# Code base for Object detection on CIFAR10 using Pytorch

This repo is the internal engine for our Object detection projects on CIFAR10 using Pytorch. Lets take a look at the components:

## [model](/model)

Here you can find code structure for all the models that we have used. Following is the list, for now:

* [Resnet18](/model/resnet.py)
* [custom_resnet](/model/custom_resnet.py)

## [utils](/utils)

Home for code related to processing and augmentation of images. Current residents:

* [gradcam.py](/utils/gradcam.py)  
  
  Gradcam code to _visually understand_ parts of the image that our model is focusing on.  
  
  _Reference:_ [Kazuto Nakashima](https://github.com/kazuto1011/grad-cam-pytorch/blob/fd10ff7fc85ae064938531235a5dd3889ca46fed/grad_cam.py)

* [utils.py](/utils/utils.py)  

  Contains following processes:  
  * Mean and Standard Deviation calculation
  * Image Transformation
  * Downloading the Dataset
  * Dataloaders
  * Display sample images from the dataset
  * Plot loss and accuracy graphs
  * Seeding
  * Display images that were misclassified

* [lr_finder.py](/utils/lr_finder.py)  

  Calculate the best _max_lr_ parameter for _OneCycleLR_.
  
## [main.py](/main.py)
  
You can find all the major orchestration code here. Currently available:  
* train
* test
* main

This repo is constantly being updated based on all the new and intersting Computer Vision tasks we do. Do visit again!

### Contributors
Abhiram Gurijala  
Arijit Ganguly  
Rohin Sequeira  
