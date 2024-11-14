import numpy
import json
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report
from tensorflow.keras import regularizers
import random

t1=time.time()
with open(r"datas2.txt","r") as fp:
    s=fp.read()
s=json.loads(s)
random.shuffle(s)

wav_fns={}

X=[]
y=[]
a = 0
b = 0
for r in s:
    wav_fns[r[0]]={
        "path":r[1],
        "mfcc":numpy.array(r[2]),
        "label": numpy.array(r[3])
    }
    for d in r[2]:
        if r[3][0] == 0:
            a += 1
        else:
                b += 1

        X.append(d)
        y.append(r[3])

X = numpy.array(X)
y = numpy.array(y)
from sklearn.model_selection import train_test_split
train_data, test_data1, train_labels, test_labels1 = train_test_split(X, y, test_size=0.30, random_state=42)
validation_data, test_data, validation_labels, test_labels = train_test_split(test_data1, test_labels1, test_size=0.30, random_state=42)

train_data = train_data.reshape((train_data.shape[0], 20, 1, 1))

test_data = test_data.reshape((test_data.shape[0], 20, 1, 1))
validation_data = validation_data.reshape((validation_data.shape[0], 20, 1, 1))

from tensorflow.keras import layers
from tensorflow.keras import optimizers

model = Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(20, 1, 1)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu', input_shape=train_data.shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(120, (2, 2), activation='relu', input_shape=train_data.shape))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))



