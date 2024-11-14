import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# print(os.environ)
# os.environ["TF_KERAS"] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["SM_FRAMEWORK"] = "tf.keras"

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
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
with open(r"datasFNN.txt","r") as fp:
    s=fp.read()
s=json.loads(s)
random.shuffle(s)

wav_fns={}
# x_train=numpy.array([])
# y_train=numpy.array([])
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
        # if r[3][0] == 0:
        #     a += 1
        # else:
        #     # if b>10422:
        #     #     break
        #     # else:
        #         b += 1
        # numpy.append(x_train, d)
        # numpy.append(y_train, r[3])
        x_train.append(d)
        y_train.append(r[3])
    # x_train.append(r[2])
    # y_train.append(r[3])
    # numpy.append(x_train,r[2])
    # numpy.append(y_train,r[3])


print("##########################################")
print(a)
print("##########################################")
print(b)
print("##########################################")
from sklearn.utils import shuffle
x_train,y_train = shuffle(x_train, y_train, random_state=0)

splite_at=int(len(x_train)/2)
x_test=numpy.array(x_train[splite_at:])
y_test=numpy.array(y_train[splite_at:])

x_train=numpy.array(x_train[:splite_at])
y_train=numpy.array(y_train[:splite_at])

# x_train = numpy.asarray(x_train).astype(numpy.float32)
# y_train = numpy.asarray(y_train).astype(numpy.float32)
# x_train=numpy.array(x_train,dtype=numpy.float)
# y_train=numpy.array(y_train,dtype=numpy.float)
# print(wav_fns)





# i1,i2=random.randint(0,len(x_train)),random.randint(0,len(y_train))
# x_train(i1),y_train(i2)=x_train(i1),y_train(i2)


scaler = preprocessing.StandardScaler()

scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train)
print("##########################################")
print(x_train_scaled)
print("##########################################")
print(y_train)
print("##########################################")
print(y_test)



print(time.time()-t1)
t1=time.time()
# exit()
# y,sr = librosa.load(r"G:\pap\UrbanSound8K\audio\fold1\7061-6-0-0.wav")#,44100*4
# mfcc = librosa.feature.mfcc(y,sr,n_mfcc=40)
# mfcc=numpy.mean(mfcc,axis=1)
# print(len(y),sr)
# print(mfcc)
print("ok",len(y_train))

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import Activation, Dense,Dropout
# import keras

# if 1==0:
#     c1,c2=0,0
#     model = models.load_model(r"model4.h5")
#     ys=model(x_train).numpy()
#     for i,y in enumerate(ys):
#         p1=numpy.argwhere(y==y.max())
#         p1=p1[0][0]
#         p2 = numpy.argwhere(y_train[i] == y_train[i].max())[0][0]
#         if p1==p2:
#             c1+=1
#         else:
#             c2+=1
#     print(c1,"/",c1+c2,c1/(c1+c2)*100)
#
#     exit()
#
# if 0==1:
#     model = models.load_model(r"g:\pap\model3.h5")
# else:



la = 0.0008

model = 0
model = Sequential()


# model.add(Dense(256, input_shape=(40,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.03))
#
# model.add(Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(la)))
# model.add(Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(la)))
# model.add(Dropout(0.03)) # you can delete this part
#



x_train_scaled = x_train_scaled.reshape(-1, 1, 40)
x_test_scaled = x_test_scaled.reshape(-1, 1, 40)
# y_train = y_train.reshape(-1, 1, 2)
# y_test = y_test.reshape(-1, 1, 2)

print(x_train_scaled.shape)
print(x_test_scaled.shape)
print(y_train.shape)
print(y_test.shape)

import tensorflow
# 2 LSTM layers
model.add(
    tensorflow.keras.layers.LSTM(256, input_shape=(1, 40), return_sequences=True))  # sequence to sequence LSTM layer
model.add(tensorflow.keras.layers.LSTM(64))  # sequence to vector

# dense layer
model.add(tensorflow.keras.layers.Dense(64, activation='relu'))
model.add(tensorflow.keras.layers.Dropout(0.3))


# model.add(Dense(10))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# print(type(x_train))
# print(x_train.shape)
# print(type(y_train))
# print(y_train.shape)
print("##########################################")
import matplotlib.pyplot as plt

history = model.fit(x_train_scaled,y_train,validation_data=(x_test_scaled, y_test),epochs=20,batch_size=32)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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
