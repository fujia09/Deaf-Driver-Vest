import os
import librosa
import numpy
import time

t1=time.time()
wav_fns={}
label_index={}

for root,dir,fns in os.walk(r"UrbanSound8K/audio"):
    for fn in fns:
        if fn[-3:]=="wav":
            wav_fns[fn]={"path":os.path.join(root,fn),
                         "mfcc":[],
                         "label":None}

ti=0
count=len(wav_fns.keys())
import csv
with open(r"UrbanSound8K/metadata/UrbanSound8K.csv") as fp:
    f_csv=csv.reader(fp)
    for row in f_csv:
        # print(row)
        # print(row[7])
        if (row[0] in wav_fns) : # and (row[7] != "car_horn")
            if row[7] == "car_horn":
                lb = numpy.array([0, 1])
            else:
                lb = numpy.array([1, 0])

            cid=int(row[6]) # classid

            wav_fns[row[0]]["label"]=lb.tolist() # used to store [1, 0] in the label section of wav_fins
            if cid not in label_index:
                label_index[cid]=row[7] # just a dictionary storing all the labels

            y, sr = librosa.load(wav_fns[row[0]]["path"],44100)  # ,44100*4  # it has a sample_rate of 44100. 44100 is the number of samples a second
            # y stores all the amplitudes and has a shape of 44100*4 since the video is 4 seconds long
            # 44100 sample rate represents 44100 data per second and in this case the data represents the amplitude of the sound wave
            use_len=int(44100/10) # Use _ len = int (44100 / 10) means that there are 4410 data in 100 ms
            use_count=int(len(y)/use_len) # Use _ count represents how many parts of the split (how many 4410 data in 100 ms)

            for i in range(use_count):
                #y2=y[i*use_len:(i+1)*use_len] # y2 stores data per 100ms
                y2=y[i*use_count:(i+1)*use_len]
                # print(len(y2)) # len(y2) == 4410 because there are 40 of 4410s
                mfcc = librosa.feature.mfcc(y2, sr, n_mfcc=40) # y2 is an audio time series
                # n_mfcc=number of MFCCs to return
                wav_fns[row[0]]["mfcc"].append(numpy.mean(mfcc, axis=1).tolist()) # axis 1 will be a mean of all the columns in each row
                # the shape of mfcc is (40, 9) and this takes the mean of each column
            ti+=1
            print(ti,"/",count)

s=[]
for k in wav_fns:
    if wav_fns[k]["label"] is None:
        print(k)

    else:
        w = wav_fns[k]
        s.append([k, w["path"], w["mfcc"], w["label"]])

print(time.time()-t1)

import json
s=json.dumps(s)
with open(r"MCARHORN.txt","w") as fp:
    fp.write(s)
