from keras.datasets import imdb
import numpy as np
from keras import layers
from keras import models
import matplotlib.pyplot as plt 
#from numpy.core.fromnumeric import argmax
import tensorflow as tf

#Loading the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
"""num_words=10000 means we only keep the top 10 000 most frequently occurring words
in the training data, rare words will be discarded. It allows us to work with vector
data of manageable size. Thus, no word index will exceed 10 000. train_data and 
test_data are lists of reviews encoded as word indices. train_labels and test_labels
are lists of 0s (for negative) and 1s (for positive)."""

#print(train_data[0])
#print(train_labels[0])

#Preparing the data
"""One can't feed lists of integers into a neural network, we must turn the lists into
tensors. Hence, we're going to encode the integer sequences into a binary matrix : each
lists is turned into a 10 000-dimentional vector of 0s except for the indices corresponding
to the integers in the sequence, which would be 1s."""

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension)) #we create an all-zero matrix of shape (len(sequences), dimension)
    for i, sequence in enumerate(sequences):
        results[i,sequence] = 1 #sets specific indices of results[i] to 1s
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
#print(x_train[0])
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

"""The input data is vectors and the labels are scalars (0s and 1s). We choose a network
with a simple stack of 3 fully connected (Dense) layers with relu activations : Dense(16,
activation = 'relu'). More precisely, there will be 3 layers : 2 intermediate layers
with 16 hidden units each, using relu as their activation function, and a third layer 
that will output the scalar prediction regarding the sentiment of the current review,
using a sigmoïd activation so as to output a probability (a score between 0 and 1
indicating how likely the sample is to have the target "1", or how likely is the review
to be positive)."""

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#Compiling the model
"""Since the model is a binary classification and since the output is a probability,
the best loss function is binary_crossentropy. For the optimizer, we'll use rmsprop."""
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#Setting aside a validation set
"""In order to monitor the model's accuracy on new data during training, we create a
validation set by setting apart 10 000 samples from the original training data."""
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#Training the model
"""Now we'll train the model for 20 epochs (20 iterations over all samples in the
x_train and y_train tensors), in mini-batches of 512 samples. Simultaneously, we'll
monitor loss and accuracy on the 10 000 samples we set appart by passing the validation
data as the validation_data argument."""
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val,y_val))

#Plotting the training and validation loss
"""Let's use matplotlib to plot the training and validation loss side by side. The results
can vary slightly due to a different random initialization of the network."""
history_dict = history.history
acc = history_dict['acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label="Training loss") #bo is for blue dot
plt.plot(epochs, val_loss_values, 'b', label="Validation loss") #b is for "solid blue line"
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
