# Lab Assignment: Introduction to Image Processing and Keras
*Objective*: This lab assignment will guide you through the basics of image processing using the Pillow library, followed by an introduction to Keras, a popular deep learning library in Python. By the end of this lab, you should be comfortable with basic image processing tasks and creating a simple neural network model with Keras.
## Part 1: Image Processing with Pillow
Let's start by loading an image and performing some basic operations on it.
1. First, import the required library:
```python
from PIL import Image
import matplotlib.pyplot as plt
```
2. Load an image (you can use any image you want, just make sure it's in the same directory as your notebook):
```python
# Replace 'image.jpg' with the name of your image file
img = Image.open('image.jpg')
img.show()
```
3. Resize the image:
```python
# We'll resize the image to be 128x128 pixels
img_resized = img.resize((128, 128))
img_resized.show()
```
4. Convert the image to grayscale:
```python
img_gray = img_resized.convert('L')
img_gray.show()
plt.imshow(img_gray,cmap = 'gray')
```
*Task*: Now, it's your turn. Load your own image, resize it, convert it to grayscale, and display it. Feel free to experiment with different sizes and color conversions.
## Part 2: Introduction to Keras
We'll start with a simple fully-connected neural network model. We'll use the MNIST dataset, a popular dataset of handwritten digits, which is built into Keras.
1. First, let's import the necessary libraries:
```python
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
```
2. Load the MNIST dataset:
```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```
3. Preprocess the images:
```python
# Normalize pixel values between 0 and 1
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
```
4. Preprocess the labels:
```python
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```
5. Build the model:
```python
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax'))
```
**OR**
```python
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(256, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(128, activation='relu', input_shape=(28 * 28,)))

model.add(Dense(10, activation='softmax'))
```
6. Compile the model:
```python
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
7. Train the model:
```python
model.fit(train_images, train_labels, epochs=5, batch_size=128)
```
8. Evaluate the model:
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
*Task*: Now, try building your own model. Try adding more layers, changing the number of units in the layers, or changing the activation function. See how these changes affect the accuracy of the model.
---
Remember, this is just an introduction. There are many more things you can do with image processing and Keras. Don't hesitate to experiment and try new things!
