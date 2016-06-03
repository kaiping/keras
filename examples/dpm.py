'''This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton

arXiv:1504.00941v2 [cs.NE] 7 Apr 201
http://arxiv.org/pdf/1504.00941v2.pdf

Optimizer is replaced with RMSprop which yields more stable and steady
improvement.

Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)
'''

# This is for running our DPM Model1 in Keras

from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
#from keras.layers import SimpleRNN
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Merge
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
import math

# batch_size = 32
batch_size = 1
# nb_classes = 10
# nb_epochs = 200
nb_epochs = 10
# hidden_units = 100

#learning_rate = 1e-6
learning_rate = 0.002 # consistent with our Model1 in singa
#clip_norm = 1.0
clip_norm = 2.5

time_steps = 50
feature_dim = 6035
dense_dim = 20
rnn_dim = 512
# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Feed in the data files by ourselves
file_x_train = open("train_data");
X_train = np.genfromtxt(file_x_train, delimiter = ",");

file_y_train = open("train_label");
Y_train = np.genfromtxt(file_y_train, delimiter = ",");

file_t_train = open("train_time");
T_train = np.genfromtxt(file_t_train, delimiter = ",");

file_x_test = open("test_data");
X_test = np.genfromtxt(file_x_test, delimiter = ",");

file_y_test = open("test_label");
Y_test = np.genfromtxt(file_y_test, delimiter = ",");

file_t_test = open("test_time");
T_test = np.genfromtxt(file_t_test, delimiter = ",");


# Reshaping operation
X_train = X_train.reshape(X_train.shape[0], time_steps, feature_dim)
X_test = X_test.reshape(X_test.shape[0], time_steps, feature_dim)

# No need for reshaping Y_train, Y_test, T_train, T_test?
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)
print('T_train shape:', T_train.shape)
print('T_test shape:', T_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')



print('Evaluate DPM Model1..')
model_time = Sequential()
model_time.add(Dense(dense_dim, input_dim=1))

model_input = Sequential()
model_input.add(GRU(rnn_dim)) # dim for output, currently no need for dropout
model_input.add(Dense(dense_dim, input_dim=rnn_dim)) # actually no need to specifiy the rnn_dim

merged = Merge([model_time, model_input], mode = 'sum')
final_model = Sequential()
final_model.add(merged)
final_model.add(Activation('tanh'))
final_model.add(Dense(1))

rmsprop = RMSprop(lr=learning_rate)
final_model.compile(loss='mean_squared_error',
              optimizer=rmsprop,
              metrics=['mean_squared_error'])

# TODO(Kaiping): How to make data. time, label correspond to each variable correctly?
final_model.fit(X_train, T_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(X_test, T_test, Y_test)) # What is "verbose=1"? verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = verbose, 2 = one log line per epoch.

scores = final_model.evaluate(X_test, T_test, Y_test, verbose=0)
print('DPM Model1 loss:', scores[0])
print('DPM Model1 test MSE:', scores[1])
print('DPM Model1 test Rooted MSE:', math.sqrt(scores[1]))
