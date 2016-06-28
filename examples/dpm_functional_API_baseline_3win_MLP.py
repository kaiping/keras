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
from keras.layers import merge
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
import math
from keras.utils.visualize_util import plot

batch_sizes = 1
nb_epochs = 10

#learning_rate = 1e-6
learning_rate = 0.002 # consistent with our Model1 in singa
#clip_norm = 1.0
clip_norm = 2.5

#time_steps = 3 # here means window number
feature_dim = 5091 # 5089 medical features + 2 demographical features
mlp_dim = 512
dense_dim = 20

# Feed in the data files by ourselves
file = open("3_win_output.txt")
all_data = np.genfromtxt(file, delimiter=",")
np.random.shuffle(all_data) # do shuffling before dividing into train & test

# No need to change for this part (RNN VS MLP should be exactly the same)
test_per = 0.1 # dividing into train & test
test_sample = int(test_per * all_data.shape[0])
test_data_part = all_data[0:test_sample,:]
train_data_part = all_data[test_sample:,:]

X_test_win1 = test_data_part[:,0:feature_dim * 1]
X_test_win2 = test_data_part[:,feature_dim * 1:feature_dim * 2]
X_test_win3 = test_data_part[:,feature_dim * 2:feature_dim * 3]
T_test = test_data_part[:,feature_dim * 3:feature_dim * 3 + 1]
Y_test = test_data_part[:,feature_dim * 3 + 1:feature_dim * 3 + 2]

X_train_win1 = train_data_part[:,0:feature_dim * 1]
X_train_win2 = train_data_part[:,feature_dim * 1:feature_dim * 2]
X_train_win3 = train_data_part[:,feature_dim * 2:feature_dim * 3]
T_train = train_data_part[:,feature_dim * 3:feature_dim * 3 + 1]
Y_train = train_data_part[:,feature_dim * 3 + 1:feature_dim * 3 + 2]

print "All data shape:", all_data.shape
print 'X_train win1 shape:', X_train_win1.shape
print 'X_train win2 shape:', X_train_win2.shape
print 'X_train win3 shape:', X_train_win3.shape
print 'Y_train shape:', Y_train.shape
print 'T_train shape:', T_train.shape

print 'X_test win1 shape:', X_test_win1.shape
print 'X_test win2 shape:', X_test_win2.shape
print 'X_test win3 shape:', X_test_win3.shape
print 'Y_test shape:', Y_test.shape
print 'T_test shape:', T_test.shape

print X_train_win1.shape[0], 'train samples\n'
print X_test_win1.shape[0], 'test samples\n'

print 'Evaluate DPM Baseline1...3 windows...CutPoint=12months...MLP (all Dense) Model'
# Feature input part
feature_input_win1 = Input(shape=(feature_dim,), name='feature_input_win1') # here no need to mention "batchsize"
feature_input_win2 = Input(shape=(feature_dim,), name='feature_input_win2')
feature_input_win3 = Input(shape=(feature_dim,), name='feature_input_win3')

# 1st set of Dense layers which take place of RNN
mlp1 = Dense(mlp_dim)(feature_input_win1)
mlp2 = Dense(mlp_dim)(feature_input_win2)
mlp3 = Dense(mlp_dim)(feature_input_win3)

# 2nd set of Dense layers before merging
dense1 = Dense(dense_dim)(mlp1)
dense2 = Dense(dense_dim)(mlp2)
dense3 = Dense(dense_dim)(mlp3)

# Delta time input part
time_input = Input(shape=(1,), name='time_input')
dense_time = Dense(dense_dim)(time_input)

# Merge these (win_num + 1 for time) inputs together
merge_output = merge([dense1, dense2, dense3, dense_time], mode='sum')

activation = Activation('tanh')(merge_output)
prediction = Dense(1)(activation)

model = Model(input=[feature_input_win1, feature_input_win2, feature_input_win3, time_input], output=[prediction])

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_squared_error'])

history = model.fit([X_train_win1, X_train_win2, X_train_win3, T_train], [Y_train], nb_epoch=nb_epochs, batch_size=batch_sizes, validation_data=([X_test_win1, X_test_win2, X_test_win3, T_test], [Y_test]))

print "Some history information\n"
print history.history
plot(model, to_file='dpm_model_3win_mlp.png')


#pass
