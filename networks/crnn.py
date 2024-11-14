
import sys
import os
import IPython
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


import random
from datetime import datetime

import tensorflow
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SpatialDropout2D,Convolution2D, GRU, Input, Dropout, Flatten, Reshape, Permute, LSTM, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Define general variables

# Set your path to the dataset
us8k_path = os.path.abspath('./UrbanSound8K')
audio_path = os.path.join(us8k_path, 'audio')
metadata_path = os.path.join(us8k_path, 'metadata/UrbanSound8K.csv')
models_path = os.path.abspath('./models')
data_path = os.path.abspath('./data')

# Ensure "channel last" data format on Keras
keras_backend.set_image_data_format('channels_last')

# Define a labels array for future use
labels = [
        'Air Conditioner',
        'Car Horn',
        'Children Playing',
        'Dog bark',
        'Drilling',
        'Engine Idling',
        'Gun Shot',
        'Jackhammer',
        'Siren',
        'Street Music'
    ]

# Pre-processed MFCC coefficients
X = np.load("data/X-mfcc.npy")
y = np.load("data/y-mfcc.npy")

print(X.shape)
# Metadata
metadata = pd.read_csv(metadata_path)

indexes = []
total = len(metadata)
indexes = list(range(0, total))

# Randomize indexes
random.shuffle(indexes)

# Divide the indexes into Train and Test
test_split_pct = 20
split_offset = math.floor(test_split_pct * total / 100)

# Split the metadata
test_split_idx = indexes[0:split_offset]
train_split_idx = indexes[split_offset:total]


# Split the features with the same indexes
X_test = np.take(X, test_split_idx, axis=0)
y_test = np.take(y, test_split_idx, axis=0)
X_train = np.take(X, train_split_idx, axis=0)
y_train = np.take(y, train_split_idx, axis=0)

# Also split metadata
test_meta = metadata.iloc[test_split_idx]
train_meta = metadata.iloc[train_split_idx]

# Print status
print("Test split: {} \t\t Train split: {}".format(len(test_meta), len(train_meta)))
print("X test shape: {} \t X train shape: {}".format(X_test.shape, X_train.shape))
print("y test shape: {} \t\t y train shape: {}".format(y_test.shape, y_train.shape))

le = LabelEncoder()
y_test_encoded = to_categorical(le.fit_transform(y_test))
y_train_encoded = to_categorical(le.fit_transform(y_train))

# How data should be structured
num_rows = 40
num_columns = 174
num_channels = 1


# Reshape to fit the network input (channel last)
X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)


# Total number of labels to predict (equal to the network output nodes)
num_labels = y_train_encoded.shape[1]

# def trainData(dataset):
#     print('TotalCount: {}'.format(totalRecordCount))
#     trainDataEndIndex = int(totalRecordCount*0.8)
#     random.shuffle(dataset)
#
#     train = dataset[:trainDataEndIndex]
#     test = dataset[trainDataEndIndex:]
#
#     print('Total training data:{}'.format(len(train)))
#     print('Total test data:{}'.format(len(test)))
#
#     # Get the data (128, 128) and label from tuple
#     X_train, y_train = zip(*train)
#     X_test, y_test = zip(*test)
#
#     # Reshape for CNN input
#     X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
#     X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])

    # One-Hot encoding for classes


print(X_train.shape)
from tensorflow.keras import backend as K
tensorflow.keras.backend.image_data_format()



inputs = Input(shape=(40, 174, 1))
x = Convolution2D(30, 3, 3, activation='relu', name='conv1')(inputs)
x = MaxPooling2D(pool_size=(3, 2), strides=(3, 2), name='pool1')(x)
x = Dropout(0.1, name='dropout1')(x)

x = Convolution2D(60, 3, 3, activation='relu', name='conv2')(x)
x = MaxPooling2D(pool_size=(3, 2), strides=(3, 2), name='pool2')(x)
x = Dropout(0.1, name='dropout2')(x)

x = Convolution2D(60, 3, 3, activation='relu', name='conv3')(x)
x = MaxPooling2D(pool_size=(4, 2), strides=(4, 2), name='pool3')(x)
x = Dropout(0.1, name='dropout3')(x)

x = Convolution2D(60, 3, 3, activation='relu', name='conv4')(x)
x = MaxPooling2D(pool_size=(4, 2), strides=(4, 2), name='pool4')(x)
x = Dropout(0.1, name='dropout4')(x)

print(x.get_shape())

x = Permute((3, 1, 2))(x)
x = Reshape((14, 60))(x)

x = GRU(30, return_sequences=True, name='gru1')(x)
x = GRU(30, return_sequences=False, name='gru2')(x)
x = Dropout(0.3, name='dropout5')(x)

output = Dense(50, activation='sigmoid', name='output')(x)

model = Model(inputs, output)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, decay=1e-8, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

# TRAINING

batch_size = 32
nb_epoch = 20

# model_filename = 'crnnmodel.pkl'

# callbacks = [
#     EarlyStopping(monitor='val_acc',
#                   patience=10,
#                   verbose=1,
#                   mode='auto'),
#
#     ModelCheckpoint(model_filename, monitor='val_acc',
#                     verbose=1,
#                     save_best_only=True,
#                     mode='auto'),
# ]
# model.load_weights('crnnmodel(61-100epoch).pkl')
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, y_test), shuffle="batch")

# model.load_weights(model_filename)

# score = model.evaluate(X_val, y_val, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# plot learning curves

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

