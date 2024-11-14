import numpy
import json
import time


t1=time.time()
with open(r"datas5.txt","r") as fp:
    s=fp.read()
s=json.loads(s)

wav_fns={}


x_train=[]
y_train=[]
for r in s:
    wav_fns[r[0]]={
        "path":r[1],
        "mfcc":numpy.array(r[2]),
        "label": numpy.array(r[3])
    }
    for d in r[2]:
        x_train.append(d)
        y_train.append(r[3])

splite_at=int(len(x_train)/2)

x_test=numpy.array(x_train[splite_at:])
y_test=numpy.array(y_train[splite_at:])

x_train=numpy.array(x_train[:splite_at])
y_train=numpy.array(y_train[:splite_at])

x_train = numpy.asarray(x_train).astype(numpy.float32)
y_train = numpy.asarray(y_train).astype(numpy.float32)

print(x_train.shape)


print(time.time()-t1)
t1=time.time()
print("ok",len(y_train))

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import Activation, Dense,Dropout

model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.03))

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))

model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

print(type(x_train))
print(x_train.shape)
print(type(y_train))
print(y_train.shape)


model.fit(x_train,y_train,epochs=10,batch_size=32)

y=model(x_test).numpy()
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


model.save(r"model_2.h5")
