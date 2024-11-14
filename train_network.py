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

x_train=[]
y_train=[]
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

        x_train.append(d)
        y_train.append(r[3])

splite_at=int(len(x_train)/2)
x_test=numpy.array(x_train[splite_at:])
y_test=numpy.array(y_train[splite_at:])

x_train=numpy.array(x_train[:splite_at])
y_train=numpy.array(y_train[:splite_at])

from sklearn.utils import shuffle
x_train,y_train = shuffle(x_train, y_train, random_state=0)

scaler = preprocessing.StandardScaler()

scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(time.time()-t1)
t1=time.time()
print("ok",len(y_train))

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import Activation, Dense,Dropout
# import keras

print(x_train_scaled.shape)
print(x_test_scaled.shape)
print(y_train.shape)
print(y_test.shape)

la = 0.0008

model = 0
model = Sequential()


model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.03))

model.add(Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(la)))
model.add(Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(la)))
model.add(Dropout(0.03)) # you can delete this part

model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(x_train_scaled,y_train,epochs=10,batch_size=32)

# model.evaluate
y=model(x_test_scaled).numpy()
i1,i2=0,0
for i in range(len(y_test)):
    p1=numpy.argwhere(y[i]==y[i].max())[0][0]
    p2=numpy.argwhere(y_test[i]==y_test[i].max())[0][0]
    if p1==p2:
        i1+=1
    i2+=1
    if i%1000==0:
        print(i,len(y_test))
print(i1/i2)


model.save(r"model_3.h5")