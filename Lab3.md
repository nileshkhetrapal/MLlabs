# Lab: Building a Language Model with TensorFlow and Python

This lab will take you through the process of creating a language model using Recurrent Neural Networks (RNNs) in TensorFlow. It will provide both theoretical and practical knowledge about language models and RNNs.

**Note**: This lab is intended to be run in Google Colab. If you're not familiar with Google Colab, you can check out this [introduction tutorial](https://colab.research.google.com/notebooks/intro.ipynb).

## Part 1: Introduction to Language Models

### What are Language Models?

A language model is a type of artificial intelligence model that provides a probability distribution over a sequence of words. It's a key component in many natural language processing tasks, including machine translation, speech recognition, and text generation.

There are various types of language models, from simple ones like Unigram and Bigram models, to more complex ones like RNN-based and Transformer-based models.

**Exercise 1.1**: Can you think of an example where language models might be used in a real-world application?

## Part 2: Understanding Recurrent Neural Networks (RNNs)

### Why are RNNs suitable for Language Modeling?

Recurrent Neural Networks (RNNs) are a type of neural network architecture that are well-suited to sequence prediction problems like language modeling because they can remember previous inputs in memory. This is especially useful in language modeling where the context (previous words) often informs the next word.

**Exercise 2.1**: Can you think of another task, apart from language modeling, where RNNs might be useful?

Here's a useful [resource](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) to learn more about RNNs and how they work.

## Part 3: Setting Up the Environment

We'll be using Python and TensorFlow for this lab. To get started, let's import the necessary libraries:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
```

**Exercise 3.1**: What are each of these libraries and functions used for? 

## Part 4: Preparing the Dataset

We'll be using a text dataset for our language model. You can use any text data you like, but for this lab, let's use 'The Complete Works of William Shakespeare'.

```python
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
```

Now, let's preprocess our data:

```python
# Create a mapping from unique characters to indices
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# Divide the data into input (x) and target (y)
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
```

**Exercise 4.1**: What is the purpose of the `char2idx` and `idx2char` dictionaries?

**Exercise 4.2**: Why do we divide the data into input and target?

## Part 5: Building the Model

Now, let's build our R

NN model. We'll use an embedding layer, a LSTM layer, and a dense layer.

```python
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=64)
```

**Exercise 5.1**: What does each layer in this model do? 

## Part 6: Training the Model

Now, let's train our model.

```python
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(dataset, epochs=10, callbacks=[checkpoint_callback])
```

**Exercise 6.1**: What is the purpose of the checkpoint directory? 

## Part 7: Generating Text with the Model

Now that our model is trained, we can use it to generate new text.

```python
def generate_text(model, start_string):
  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperature results in more predictable text.
  # Higher temperature results in more surprising text.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string="ROMEO: "))
```

**Exercise 7.1**: What is the role of temperature in the text generation process?

## Part 8: Conclusion and Further Reading

Congratulations on completing this lab! You have now built a simple language model using RNNs in TensorFlow. This model can be used to generate new text that is similar to the training data.

For further exploration, you can try adjusting the model architecture, using different types of RNN layers (like GRU), or even training a transformer-based language model.

Here are some additional resources for further reading:

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
-

 [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Deep Learning for NLP](https://www.tensorflow.org/tutorials/text/text_classification_rnn) by TensorFlow

**Exercise 8.1**: Think about another application where you can use this language model. 

**Final Project**: Try to apply what you've learned to create a language model on a different dataset. It could be news articles, tweets, or any other text data of your interest. Evaluate the generated text and discuss the findings.
