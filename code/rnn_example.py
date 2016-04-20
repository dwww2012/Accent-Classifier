from __future__ import print_function
import numpy as np

from keras.optimizers import SGD
#from keras.utils.visualize_util import plot

np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM
#from SpeechResearch import loadData
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

batch_size = 25
hidden_units = 10
nb_classes = 3
print('Loading data...')
X = np.load('top_3_100_split_mfcc.npy')
y = np.load('top_3_100_split_y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('Build model...')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# print(batch_size, 99, X_train.shape[2])
# print(X_train.shape[1:])
# print(X_train.shape[2])
model = Sequential()

batch_input_shape= (batch_size, X_train.shape[1], X_train.shape[2])

model.add(LSTM(64, return_sequences=True, stateful=False,
               batch_input_shape= (batch_size, X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64, return_sequences=True, stateful=False))
model.add(LSTM(64, stateful=False))

#
# model.add(TimeDistributedDense(input_dim=hidden_units, output_dim=nb_classes))
# model.add(Activation('softmax'))

model.add(Dropout(.25))
model.add(Dense(nb_classes, activation='softmax'))

# try using different optimizers and different optimizer configs
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=5, validation_data=(X_test, Y_test))

y_pred=model.predict_classes(X_test, batch_size=batch_size)
print(classification_report(y_test, y_pred))
# score, acc = model.evaluate(X_test, Y_test,
#                             batch_size=batch_size,
#                             show_accuracy=True)
# print('Test score:', score)
# print('Test accuracy:', acc)
