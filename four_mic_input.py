import pyaudio

p = pyaudio.PyAudio()

print(p.get_device_count())
for i in range(p.get_device_count()):
    info1 = p.get_device_info_by_index(i)
    print(info1)
    if info1["maxInputChannels"] == 0:
        continue
    if "USB" not in info1["name"]:
        continue
    info = {
        "index": info1["index"],
        "structVersion": info1["structVersion"],
        "hostApi": info1["hostApi"],
        "defaultSampleRate": info1["defaultSampleRate"],
        "InC": info1["maxInputChannels"],
        "OutC": info1["maxOutputChannels"],
        "InputLatency": "%.4f~%.4f" % (info1["defaultLowInputLatency"], info1["defaultHighInputLatency"]),
        "OutputLatency": "%.4f~%.4f" % (info1["defaultLowOutputLatency"], info1["defaultHighOutputLatency"]),
        " " * 30 + "name" + " " * 30: info1["name"]
    }

    if i == 0:
        for k in info:
            print(str("%" + str(len(k)) + "s | ") % k, end="")
        print()
    for k in info:
        print(str("%" + str(len(k)) + "s | ") % str(info[k])[:len(k)], end="")
    print()

try:
    import numpy
    from tensorflow import keras
    import librosa

    model = keras.models.load_model(r"M.h5")
    # model=keras.models.load_model(r"g:\pap\model10.h5")
    print("model loaded,press any key to continue")
    input()
except:
    exit()

label_index = {3: 'dog_bark', 2: 'children_playing', 1: 'car_horn', 0: 'air_conditioner', 9: 'street_music',
               6: 'gun_shot', 8: 'siren', 5: 'engine_idling', 7: 'jackhammer', 4: 'drilling'}
# label_index={3: '狗吠声', 2: '儿童玩耍', 1: '汽车喇叭', 0: '冷气机', 9: '街头音乐', 6: '枪射击', 8: '警笛', 5: '发动机空转', 7: '手持式凿岩机', 4: '钻孔'}
for k in label_index:
    print(k, label_index[k])

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024 * 16
CHUNK = 44100

import struct

bComplete = False


def cb1(in_data, frame_count, time_info, status):
    ints = struct.unpack('h' * CHUNK, in_data)
    if max(ints) > 2000:
        pass
    if bComplete:
        return (in_data, pyaudio.paComplete)
    else:
        return (in_data, pyaudio.paContinue)


import time

ot = time.time()
oinfo = None
tt = time.time()

bRecord = False
if bRecord:
    import wave

    wf = wave.open(r"out_wav2.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

beelist = [[],[],[],[]]
highestints = 0
highestclass = ""
highestmicrophoneid = -1
highestconfidence = 0

def cb2(microphone_id, in_data, frame_count, time_info, status):
    if bRecord:
        wf.writeframes(in_data)
        print(frame_count)

    global ot, oinfo, tt, beelist, highestints, highestclass, highestconfidence, highestmicrophoneid

    # print("-------------",status,"cost time :",time.time()-ot,frame_count)
    ot = time.time()
    if oinfo:
        # print("\t",time_info["input_buffer_adc_time"]-oinfo["input_buffer_adc_time"],
        #      time_info["current_time"]-oinfo["current_time"])
        tt += time_info["current_time"] - oinfo["current_time"]
        # print("\t",time.time()-tt)
    oinfo = time_info

    ints = struct.unpack('h' * CHUNK, in_data)
    print(max(ints), min(ints))

    if max(ints) > 1200:
        # print(2,max(ints),min(ints))

        y = numpy.array(ints, dtype=numpy.float64) / 32768
        mfcc = librosa.feature.mfcc(y, 44100, n_mfcc=40)
        mfcc = numpy.array([numpy.mean(mfcc, axis=1)])
        result = model(mfcc).numpy()[0]
        print(result)

        if result.max() > 0.9:
            idx = numpy.argwhere(result == result.max())[0][0]
            if max(ints) > highestints:
                highestints = max(ints)
                highestclass = idx
                highestconfidence = result.max()
                highestmicrophoneid = microphone_id

            # beelist[microphone_id].append((result.max(), idx, max(ints)))
            # print(idx, label_index[int(idx)], "%.2f" % result.max())
    if microphone_id == 3:
        print(highestclass, label_index[int(highestclass)], "%.2f" % highestconfidence.max(), highestmicrophoneid)
    # reset variables
        highestints = 0
        highestclass = ""
        highestmicrophoneid = -1
        highestconfidence = 0




    if bComplete:
        if bRecord:
            wf.close()
        return (in_data, pyaudio.paComplete)
    else:
        return (in_data, pyaudio.paContinue)


'''
in1=p.open(format=FORMAT,
           channels=CHANNELS,
           rate=RATE,
           input=True,
           input_device_index=0,
           frames_per_buffer=CHUNK,
           stream_callback=cb1)'''


# if they all predict the same thing, select the sensor with the loudest INPUT
def callback1(in_data, frame_count, time_info, status):
    cb2(0, in_data, frame_count, time_info, status)


def callback2(in_data, frame_count, time_info, status):
    cb2(1, in_data, frame_count, time_info, status)

def callback3(in_data, frame_count, time_info, status):
    cb2(2, in_data, frame_count, time_info, status)


def callback4(in_data, frame_count, time_info, status):
    cb2(3, in_data, frame_count, time_info, status)

in1 = p.open(format=FORMAT,
             channels=CHANNELS,
             rate=RATE,
             input=True,
             input_device_index=0,
             frames_per_buffer=CHUNK,
             stream_callback=callback1)
in2 = p.open(format=FORMAT,
             channels=CHANNELS,
             rate=RATE,
             input=True,
             input_device_index=1,
             frames_per_buffer=CHUNK,
             stream_callback=callback2)
in3 = p.open(format=FORMAT,
             channels=CHANNELS,
             rate=RATE,
             input=True,
             input_device_index=2,
             frames_per_buffer=CHUNK,
             stream_callback=callback3)
in4 = p.open(format=FORMAT,
             channels=CHANNELS,
             rate=RATE,
             input=True,
             input_device_index=3,
             frames_per_buffer=CHUNK,
             stream_callback=callback4)

input()
bComplete = True
import time

# while in1.is_active():
#    time.sleep(0.1)
while in2.is_active():
    time.sleep(0.1)