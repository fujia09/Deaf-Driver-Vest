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
        if (row[0] in wav_fns) :
            if row[7] == "car_horn":
                lb = numpy.array([0, 1])
            else:
                lb = numpy.array([1, 0])
            cid=int(row[6]) # classid
            wav_fns[row[0]]["label"]=lb.tolist() # used to store [1, 0] in the label section of wav_fins
            if cid not in label_index:
                label_index[cid]=row[7] # just a dictionary storing all the labels

            y, sr = librosa.load(wav_fns[row[0]]["path"])  # ,44100*4  # it has a sample_rate of 44100. 44100 is the number of samples a second
            D = librosa.stft(y)
            spec = librosa.power_to_db(numpy.abs(D) ** 2, ref=numpy.median)
            wav_fns[row[0]]["mfcc"].append(spec.tolist())

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
with open(r"datas4.txt","w") as fp:
    fp.write(s)