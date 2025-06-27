#from __future__ import print_function
import gc
import keras
import theano
import random
import numpy as np
import pandas as pd
import datetime as dt
import shutil, os, sys
import tensorflow as tf
#
from numpy import genfromtxt
from keras import optimizers
from keras import regularizers
from keras import initializers
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.constraints import maxnorm
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Activation, Embedding, GRU, RepeatVector, SpatialDropout1D, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, LSTM, BatchNormalization, GlobalMaxPooling1D
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight


def shuffleData(X, y):
	index = [i for i in range(len(X))]
	random.shuffle(index)
	X = X[index]
	y = y[index]
	return X, y


percentage = sys.argv[1]
tar_rep_id = sys.argv[2]
# gpu_id = sys.argv[3]
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
# config = tf.ConfigProto()
# config.allow_soft_placement = True
# config.gpu_options.per_process_gpu_memory_fraction=0.7
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
#
theano.config.openmp = True
np.set_printoptions(threshold=sys.maxsize)
#
nb_epoch = 50
num_classes = 2
max_seq_len = 1820

mulu_train = "p%s_trte/" % percentage
mulu_models = "MODELs/"
mulu_logs = "LOGs/"
if not os.path.exists(mulu_models):
	os.makedirs(mulu_models)
if not os.path.exists(mulu_logs):
	os.makedirs(mulu_logs)
#
tar_mulu_model = "%s/MODEL_p%s/" % (mulu_models, percentage)
if not os.path.exists(tar_mulu_model):
	os.makedirs(tar_mulu_model)

tar_prefix = "R%s_p%s" % (tar_rep_id, percentage)
tar_log_file = "%sLOG_%s" % (mulu_logs, tar_prefix)
fout = open(tar_log_file, 'w', 0)

print('Loading data...')
infile_train = '%sCoding_Train_%s' % (mulu_train, tar_rep_id)
infile_test = '%sCoding_Test_%s' % (mulu_train, tar_rep_id)
print(infile_train)
print(infile_test)
times1 = dt.datetime.now()

train_data = pd.read_csv(infile_train, index_col=False, header=None)
test_data = pd.read_csv(infile_test, index_col=False, header=None)
print(train_data.shape)
print(test_data.shape)
y_train_ori = train_data[0]
x_train_ori = train_data[1]
y_test_ori = test_data[0]
x_test_ori = test_data[1]

x_test, y_test = [], []
for pi in x_test_ori:
	nr = pi.split(' ')[0:-1]
	ndata = map(int, nr)
	x_test.append(ndata)
x_test = np.array(x_test)
for pi in y_test_ori:
	nr = pi.split(' ')[0]
	ndata = int(nr)
	y_test.append(ndata)
y_test = np.array(y_test)

x_train, y_train = [], []
for pi in x_train_ori:
	nr = pi.split(' ')[0:-1]
	ndata = map(int, nr)
	x_train.append(ndata)
x_train = np.array(x_train)
for pi in y_train_ori:
	nr = pi.split(' ')[0]
	ndata = int(nr)
	y_train.append(ndata)
y_train = np.array(y_train)
print(x_train.shape)
print(x_test[0], y_test)

times2 = dt.datetime.now()
print('Time spent: %s' % str(times2 - times1))

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
y_real = np.argmax(y_test, axis=1)
x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len, padding='post', truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=max_seq_len, padding='post', truncating='post')
print(x_train.shape)


Dropout1 = 0.0309180687648
Dropout2 = 0.0452235535473
Dropout3 = 0.274720330055
Dropout4 = 0.174916356643
Dropout5 = 0.494506347253
Lr1 = 0.00679460941962
Lr2 = 0.00570815572816
kernel_size1 = 15
kernel_size2 = 15
filter1 = 32
filter2 = 32
Batch_size1 = 64
Lstm1 = 64
Pool_size1 = 8
# Optimizer1 = 'adam'
#
Dropout_0 = Dropout1
Dropout_1 = Dropout2
Dropout_2 = Dropout3
Dropout_3 = Dropout4
Dropout_4 = Dropout5
Lr1_0 = Lr1
Lr1_1 = Lr2
kernel_size1_0 = kernel_size1
kernel_size1_1 = kernel_size2
filter1_0 = filter1
filter1_1 = filter2
Batch_size1 = Batch_size1
filter1_2 = Lstm1
Pool_size1 = Pool_size1
Optimizer1 = Adam(amsgrad=True)  # Optimizer1

#
model_filepath= "%sBest_model_%s.h5" % (tar_mulu_model, tar_prefix)
#
model = Sequential()
model.add(Embedding(21, 21, input_length=max_seq_len))
model.add(BatchNormalization())
model.add(Dropout(Dropout_0))
#
model.add(Conv1D(filters=filter1_0, kernel_size=kernel_size1_0, activation='relu', padding='same', strides=1,
				 kernel_regularizer=regularizers.l2(Lr1_0), kernel_initializer='random_uniform',
				 kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Dropout(Dropout_1))
#
model.add(Conv1D(filters=filter1_1, kernel_size=kernel_size1_1, activation='relu', padding='same', strides=1,
				 kernel_regularizer=regularizers.l2(Lr1_1), kernel_initializer='random_uniform',
				 kernel_constraint=maxnorm(3),bias_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Dropout(Dropout_2))
#
model.add(MaxPooling1D(pool_size=Pool_size1))
model.add(BatchNormalization())
model.add(Dropout(Dropout_3))
#
model.add(LSTM(filter1_2))
model.add(BatchNormalization())
model.add(Dropout(Dropout_4))
#
model.add(Dense(num_classes, activation='sigmoid'))
#
print(model.summary())
checkpoint = ModelCheckpoint(model_filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print('Training')
for re in range(nb_epoch):
	tar_re = re+1
	print('Epoch: %s / %s' % (tar_re, nb_epoch))
	x_train, y_train = shuffleData(x_train,y_train)
	model.compile(loss='binary_crossentropy',optimizer=Optimizer1, metrics=['accuracy'])
	Result=model.fit(x_train, y_train, epochs=1, callbacks=callbacks_list,
					 batch_size=Batch_size1, validation_data=(x_test, y_test),
					 shuffle=True, class_weight='auto', verbose = 1)
	loss_and_metrics_train = model.evaluate(x_train, y_train)
	loss_and_metrics_test = model.evaluate(x_test, y_test)
	print >>fout, "Train_Test " + str(tar_re) +" metrics ",loss_and_metrics_train, loss_and_metrics_test
	print(loss_and_metrics_train)
	print(loss_and_metrics_train[1])
del model


model = load_model(model_filepath)
loss_and_metrics = model.evaluate(x_test, y_test)
y_pred_ori = model.predict(x_test)
y_pred = np.argmax(y_pred_ori, axis=1)  # Convert one-hot to index
#
print >>fout, "Final_metrics_Test : %s" % loss_and_metrics
#
loss_and_metrics_train = model.evaluate(x_train, y_train)
print >>fout, "Final_metrics_Train : %s\n\n" % loss_and_metrics_train
print >>fout, classification_report(y_real, y_pred, digits=6)
#
auc = roc_auc_score(y_test.flatten(), y_pred_ori.flatten())
print >>fout, "AUC"
print >>fout, auc
#
fout.close()
