
from __future__ import print_function
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.layers.noise import GaussianNoise
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils


# set parameters:
test_dim = 999
maxlen = 100
batch_size = 50
nb_filter = 512
filter_length_1 = 100
filter_length_2 = 30
filter_length_3 = 15
hidden_dims = 10
nb_epoch = 5
nb_classes = 3

print('Loading data...')
X = np.load('top_3_100_split_mfcc.npy')
y = np.load('top_3_100_split_y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
'''
xts = X_train.shape
X_train = np.reshape(X_train, (xts[0], xts[1], 1))
xtss = X_test.shape
X_test = np.reshape(X_test, (xtss[0], xtss[1], 1))
yts = y_train.shape
y_train = np.reshape(y_train, (yts[0], 1))
ytss = y_test.shape
y_test = np.reshape(y_test, (ytss[0], 1))
'''
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# print('Pad sequences (samples x time)')
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
# model.add(Dropout(0.25))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length_1,
                        input_shape=(test_dim, 13),
                        init = 'glorot_normal',
                        border_mode='valid',
                        activation='relu'
                        ))
# we use standard max pooling (halving the output of the previous layer):
model.add(BatchNormalization())


model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length_2,          
			border_mode='valid',
                        activation='relu'
                        ))

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length_2,
                        border_mode='valid',
                        activation='relu'
                        ))

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length_2,
                        border_mode='valid',
                        activation='relu'
                        ))

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_length=2))

model.add(Dropout(.1))

model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length_2,
                        border_mode='valid',
                        activation='relu'
                        ))

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_length=2))

model.add(Dropout(.1))

model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length_3,
                        border_mode='valid',
                        activation='relu'
                        ))

model.add(BatchNormalization())

model.add(MaxPooling1D(pool_length=2))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
#model.add(Dense(hidden_dims, init = 'glorot_normal'))
#model.add(Dropout(0.2))
#model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics = ["accuracy"])
model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=nb_epoch,  verbose=1,
          validation_data=(X_test, Y_test))

y_pred = model.predict_classes(X_test)
print(classification_report(y_test, y_pred))
