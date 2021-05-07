---
title: Tricks for Training a State-of-the-art Deep Learning model for Computer Vision in Fast.AI
author: Max van Dijck
date: 2021-05-07 21:00:00 +1200
categories: [Tutorial]
tags: [DeepLearning, FastAI]
---

There exists many ways to easily and effectively improve the accuracy of a neural network simply by making small changes to the data, training pipeline or how we run inference. This post aims to outline the most common, effective techniques, how they work and how they’re implemented in fast.ai.

## Data
### Augmentation

Data augmentation is the act of taking your training data and creating brand new images by a means of warping, flipping, rotating and more. Not only does this create a larger dataset but provides more unique images meaning the model can better generalize across the classes it is learning.
Fast.ai already has [**a number of**](https://docs.fast.ai/vision.augment.html) image augmentation methods built in such as rotation, cropping and zooming. These can be easily passed to a datablock using the aug_transforms function like so:

```python
#Generic data-block with aug_transforms
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                   get_items=get_image_files, 
                   splitter=RandomSplitter(valid_pct=0.2, seed=42),
                   get_y=parent_label,
                   item_tfms=Resize(256),
                   batch_tfms=aug_transforms(mult = 1, #aug_transforms here
                                             do_flip = True,
                                             flip_vert = False,
                                             max_rotate = 10,
                                             min_zoom = 1,
                                             max_zoom = 1.1,
                                             max_lighting = 0.2,
                                             max_warp = 0.2)
                  )
```
If creating augmentations through fast.ai isn’t your thing I’d recommend using [**Albumentations**](https://albumentations.ai/). Not only does Albumentations provide all the image transforms you could ever dream of, it’s pipeline is extremely easy to understand and it even works with image masks / segmentation:

```python
#Import Albumentations
import albumentations as A
from PIL import Image
import numpy as np

#Define transform pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

#Open image using PIL and convert to numpy array
image = Image.open('image.jpg')
image = np.array(image)

#Apply transforms to image and then extract it from the python dictionary
transformed_image = transform(image=image)['image']
```

### Normalization

When training models it helps if all images that are fed to your model all have the same range of input values, this is where normalization comes in handy, enabling your model to train on values that, generally, have a mean of 0 and standard deviation of 1 means that the model does not have to account for abnormalities in your data, this is especially important when using a pretrained model because the model will see something very different from what you intended if your data is of a different range than what was used to originally train it.
Normalization is also easy to implement using fast.ai, as easy as passing in a transform to the batch_tfms:

```python
#Generic data-block
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                   get_items=get_image_files, 
                   splitter=RandomSplitter(valid_pct=0.2, seed=42),
                   get_y=parent_label,
                   item_tfms=Resize(256),
                   batch_tfms=[*aug_transforms(mult = 1),
                              Normalize.from_stats(*imagenet_stats)]#Normalization
                  )
```

### Mixup

Mixup is a data augmentation technique that creates an image by taking a linear combination of two other images, this technique aids in generalization in situations where the labels are one-hot encoded. The pixels from one image are weighted then combined with another image, then the onehot encoded targets are adjusted to match those weights accordingly, to clarify here is a step by step process in pseudocode:

```python
#Pseudo-code for mixup
image2,target2 = dataset[randint(0,len(dataset)] #Select a second image
t = random_float(0.5,1.0) #random weight
new_image = t * image1 + (1-t) * image2 #Image combination
new_target = t * target1 + (1-t) * target2 #Label combination
```

Here is how you can add Mixup to a model in fast.ai:

```python
#Add mixup when we call our learner
learn = Learner(dls, model, 
                loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy, 
                cbs=MixUp()#Mixup
               )
```
## Training
### Progressive Resizing

When model training efficiency is what you want, progressive resizing is what you need. Progressive resizing consists of beginning training with smaller images and scaling up to large images. Spending most of the time training with smaller images keeps training time low and completing training with the larger images makes the final accuracy higher. Essentially you are creating your own transfer learning model that can learn all the important features of your dataset from smaller images and then can be retrained to classify larger images. Progressive resizing is also a form of data augmentation hence you should see better generalization.
This is how progressive resizing can be implemented using fast.ai:

```python
#Function makes a new dblock with a desired batch size and image size
#returns a dataloaders object
def get_dls(bs, size):
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=[*aug_transforms(size=size, min_scale=0.75),
                               Normalize.from_stats(*imagenet_stats)])
    return dblock.dataloaders(path, bs=bs)

#Create a learner
learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLoss(), 
                metrics=accuracy)
#Create a dataloaders with 128 batch size and 128 image size
dls = get_dls(128, 128)
#Train three cycles on the 128 sized images
learn.fit_one_cycle(3, 3e-3)
#Create a new dataloaders with smaller batch size and larger image size
learn.dls = get_dls(64, 256)
#Train three more cycles on the larger images
learn.fine_tune(3, 1e-3)
```

### Learning Rate Finder

Finding an appropriate learning rate for a model can often be the most important but confusing task in training a neural network, luckily, fast.ai provides a specific function that measures lots of different learning rates and outputs a graph that provides practitioners an easy way to find that appropriate learning rate. A good rule of thumb is to take the lr_min and divide by 10, here’s how to implement it in fast.ai:

```python
#Initialise a model
learn = cnn_learner(dls, resnet34, metrics=error_rate)
#call the learning rate finder
learn.lr_find()
```

### Discriminitive Learning Rates

Discriminative learning rates are important when transfer learning, they ensure that you retain the high quality edge detection and gradient detection in early layers by lowering the learning rate for them, the learning rate gradually increases layer by layer until your final layer. This is also useful to use after progressive resizing and can help your model gain a small amount of accuracy, this is how it’s implemented in fast.ai:

```python
#Initialise a model
learn = cnn_learner(dls, resnet34, metrics=error_rate)
#Use the slice command to define a minimum and maximum learning rate
#Each layer will be assigned a learning rate
#which is multiplicatively equidistant throughout that range
learn.fit_one_cycle(10, lr_max=slice(1e-6,1e-4))
```
