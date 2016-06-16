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

from keras.models import Model
from keras.layers import Dense, Activation
from keras.layers import Input
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import merge
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
import math

batch_sizes = 1
nb_epochs = 10

#learning_rate = 1e-6
learning_rate = 0.002 # consistent with our Model1 in singa
#clip_norm = 1.0
clip_norm = 2.5

time_steps = 3 # here means window number
feature_dim = 5091 # 5089 medical features + 2 demographical features
dense_dim = 20
rnn_dim = 512
# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Feed in the data files by ourselves
file = open("3_win_output.txt")
all_data = np.genfromtxt(file, delimiter=",")
np.random.shuffle(all_data) # do shuffling before dividing into train & test
test_per = 0.1 # dividing into train & test
test_sample = int(test_per * all_data.shape[0])
test_data_part = all_data[0:test_sample,:]
train_data_part = all_data[test_sample:,:]

X_test = test_data_part[:,0:15273]
T_test = test_data_part[:,15273:15274]
Y_test = test_data_part[:,15274:15275]

X_train = train_data_part[:,0:15273]
T_train = train_data_part[:,15273:15274]
Y_train = train_data_part[:,15274:15275]

print "All data shape:", all_data.shape
print 'X_train shape:', X_train.shape
print 'Y_train shape:', Y_train.shape
print 'T_train shape:', T_train.shape

print 'X_test shape:', X_test.shape
print 'Y_test shape:', Y_test.shape
print 'T_test shape:', T_test.shape

print X_train.shape[0], 'train samples\n'
print X_test.shape[0], 'test samples\n'

X_train = X_train.reshape(X_train.shape[0], time_steps, feature_dim)
X_test = X_test.reshape(X_test.shape[0], time_steps, feature_dim)

print 'Evaluate DPM Baseline1...3 windows...CutPoint=12months...LSTM Model'
# Feature input part
feature_input = Input(shape=(time_steps,feature_dim), name='feature_input')
#print('feature_input shape: ', feature_input.shape)
lstm_out = LSTM(rnn_dim, input_dim=feature_dim, input_length=time_steps)(feature_input)
#print('GRU shape: ', gru_out.shape)
dense1 = Dense(dense_dim)(lstm_out)
#print('dense1 shape: ', dense1.shape)

# Delta time input part
time_input = Input(shape=(1,), name='time_input')
#print('time input shape: ', time_input.shape)
dense2 = Dense(dense_dim)(time_input)
#print('dense2 shape: ', dense2.shape)

# Merge these two inputs together
merge_output = merge([dense1, dense2], mode='sum')
#print('merge shape: ', merge_output.shape)

activation = Activation('tanh')(merge_output)
#print('activation shape: ', activation.shape)
prediction = Dense(1)(activation)
#print('prediction shape: ', prediction.shape)

model = Model(input=[feature_input, time_input], output=[prediction])

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_squared_error'])

history = model.fit([X_train, T_train], [Y_train], nb_epoch=nb_epochs, batch_size=batch_sizes, validation_data=([X_test, T_test], [Y_test]))

print "Some history information\n"
print history.history


#pass
