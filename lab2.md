---

# Lab Assignment: Image Classification with Keras and CNNs

**Objective**: This lab assignment will introduce you to Convolutional Neural Networks (CNNs) and guide you through building a CNN model using Keras for image classification. The goal here is to use the CIFAR-10 dataset which contains 60,000 color images in 10 classes. By the end of this lab, you should be comfortable with building and training a CNN using Keras.

**Note**: This lab includes some advanced concepts in deep learning, but don't worry! We'll walk you through it all step by step.

## Part 1: Understanding CNNs

Convolutional Neural Networks (CNNs) are a type of deep learning model that are especially good at processing grid-like data, such as images. A CNN processes an image by applying filters to the image to identify various features such as edges, shapes, textures, etc., which are then used for classification or other tasks.

Before we dive in, read this [simple guide on CNNs](https://brohrer.github.io/how_convolutional_neural_networks_work.html) for an overview and answer the following questions:

1. What is the main difference between a regular fully connected layer and a convolutional layer in a neural network?
2. What is a feature map?
3. What are pooling layers and why are they useful in CNNs?

## Part 2: Building a CNN with Keras

Now, let's put what you've learned into practice by building a CNN to classify images from the CIFAR-10 dataset.

1. First, let's import the necessary libraries:

```python
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
```

2. Load the CIFAR-10 dataset:

```python
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
```

3. Preprocess the images:

```python
# Normalize pixel values between 0 and 1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
```

4. Preprocess the labels:

```python
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

5. **(Advanced)** Build the model. We will use a basic CNN structure with two convolutional layers, each followed by a max pooling layer, and then a fully connected layer:

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

**Think like you're 5**: We're building a tower of blocks (layers). First, we add blocks that look for simple shapes in the image (Conv2D). Then, we add blocks that take the biggest number from a small group (MaxPooling2D) to make the image smaller. We do this twice. Then, we unroll the image into a long line (Flatten) and have our regular blocks (Dense) make the final decision on what the image is.

6. Compile the model:

```python
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
``

## Part 3: Training the Model

With our model ready to go, it's time to train it with our training data.

```python
model.fit(train_images, train_labels, epochs=10, batch_size=64)
```

This will train the model for 10 epochs, with a batch size of 64. An epoch is a full pass through the entire training dataset, and a batch is a subset of the training data. The model weights are updated after each batch.

Questions:

1. What is an epoch in the context of training a neural network?
2. What is the purpose of the batch size in training a neural network?
3. What would happen if we use too large or too small batch sizes?

## Part 4: Evaluating the Model

Now that our model is trained, we can evaluate its performance on the test dataset.

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```

Questions:

1. Why do we need a separate test dataset?
2. What does the accuracy tell us about our model?
3. What other metrics can we use to evaluate our model and why might they be useful?

## Part 5: Making Predictions

Finally, we can use our trained model to make predictions on new data.

```python
predictions = model.predict(test_images)
```

This will return an array of 10 numbers. These numbers are the probabilities that the image corresponds to each of the 10 classes.

Questions:

1. What is the output of the `model.predict()` function?
2. How can we interpret the output of the prediction?

Congratulations, you've just built and trained a CNN to classify images using Keras!

## References:
- [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)【12†source】【13†source】【14†source】【15†source】【16†source】【17†source】【18†source】【19†source】
- [Keras Documentation](https://keras.io/)
