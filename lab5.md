# Lab: Understanding Convolutional Neural Networks (CNN) Configurations

In this lab, we'll delve into various popular configurations of Convolutional Neural Networks (CNNs), understand their structures, learn how to implement them using TensorFlow, and analyze their applicability to different types of problems.

## Part 1: Basic CNN

Before we dive into complex architectures, let's start with a simple CNN with just a few layers.

A basic CNN consists of a stack of Convolutional, ReLU (Rectified Linear Unit), and MaxPooling layers, followed by Fully Connected layers.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# load data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# define the CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# add dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```

[Reference](https://www.tensorflow.org/tutorials/images/cnn)

**Questions**:

1. What role do the Convolutional layers, ReLU, and MaxPooling layers play in a CNN?
2. What effect does the order of layers in a CNN have on its performance and why?

## Part 2: LeNet-5

LeNet-5, from "Gradient-Based Learning Applied to Document Recognition", is a very efficient convolutional neural network for handwritten character recognition.

```python
# LeNet-5
model = models.Sequential()
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32,32,1), padding='same')) #C1
model.add(layers.AveragePooling2D()) #S2
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')) #C3
model.add(layers.AveragePooling2D()) #S4
model.add(layers.Flatten()) #Flatten
model.add(layers.Dense(120, activation='tanh')) #C5
model.add(layers.Dense(84, activation='tanh')) #F6
model.add(layers.Dense(10, activation='softmax')) #Output layer
```

[Reference](https://engmrk.com/lenet-5-a-classic-cnn-architecture/)

**Questions:**

1. How is the structure of LeNet-5 different from our basic CNN, and why might these differences be beneficial?
2. Why do we use 'tanh' activation instead of 'relu' in LeNet-5?

## Part 3: AlexNet

AlexNet, the winner of ILSVRC (

ImageNet Large Scale Visual Recognition Challenge) 2012, is largely responsible for the popularity of CNNs today.

```python
# AlexNet
model = models.Sequential()
model.add(layers.Conv2D(96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)))
model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(layers.Conv2D(256, kernel_size=(5,5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(layers.Conv2D(384, kernel_size=(3,3), activation='relu', padding='same'))
model.add(layers.Conv2D(384, kernel_size=(3,3), activation='relu', padding='same'))
model.add(layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1000, activation='softmax'))
```

[Reference](https://engmrk.com/alexnet-implementation-using-keras/)

**Questions:**

1. How does the depth of AlexNet (number of layers) compare to previous networks we've built? What are the advantages and disadvantages of having a deeper network?
2. What's the purpose of using padding in the convolutional layers?

## Part 4: VGGNet

VGGNet, the runner-up in ILSVRC 2014, is known for its simplicity and depth. VGGNet consists of 16 convolutional layers and is very appealing because of its very uniform architecture.

[Reference and VGG16 model code](https://neurohive.io/en/popular-networks/vgg16/)

**Questions:**

1. How does VGGNet ensure that it has a much deeper architecture than previous models without increasing the complexity of model design?
2. What are the trade-offs between model complexity (depth, number of parameters etc.) and computational efficiency?

## Part 5: ResNet

ResNet, the winner of ILSVRC 2015, uses "shortcut connections" to solve the vanishing gradient problem allowing the construction of networks that are much deeper than previously possible.

[Reference and ResNet model code](https://neurohive.io/en/popular-networks/resnet/)

**Questions:**

1. How do the shortcut connections in ResNet solve the vanishing gradient problem?
2. When might you choose to use ResNet over VGGNet or AlexNet?

For all the parts, remember to compile, train and test the model as shown in Part 1.
