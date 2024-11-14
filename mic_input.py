import pyaudio
import time
import threading

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

    model = keras.models.load_model(r"MCARHORN.h5")
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
CHUNK = int(44100 * 0.5 / 10) # change this to make the frame rate slower

import struct


class XRecord:
    def __init__(self, input_device_index=0):
        self.datas=[]
        self.stream_hd = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                input_device_index=input_device_index,
                                frames_per_buffer=CHUNK,
                                stream_callback=self.callback)
        self.data_locker = threading.Lock()
        self.mic_id = input_device_index
        self.result_buffer = numpy.zeros((10, 2), dtype=numpy.float64)  # 缓存10组，每组10个声音概率
        self.result_p = 0
        self.recv_sound = {
            "id": 0,  #
            "rate": 0,  #
            "strength": 0  #
        }
        self.bNeedExit = False

    def callback(self, in_data, frame_count, time_info, status):
        self.data_locker.acquire()
        self.datas.append([in_data, frame_count, time_info, status])
        self.data_locker.release()

        if self.bNeedExit:
            return (in_data, pyaudio.paComplete)
        else:
            return (in_data, pyaudio.paContinue)

    def callback2(self):
        self.data_locker.acquire()
        if len(self.datas)==0:
            self.data_locker.release()
            return False
        in_data, frame_count, time_info, status=self.datas.pop(0)
        self.data_locker.release()



        ints = struct.unpack('h' * CHUNK, in_data)
        # print(max(ints),min(ints))
        print(max(ints))
        if max(ints) > 1200:
            # print(2,max(ints),min(ints))

            y = numpy.array(ints, dtype=numpy.float64) / 32768

            mfcc = librosa.feature.mfcc(y, 44100, n_mfcc=40)
            mfcc = numpy.array([numpy.mean(mfcc, axis=1)])
            result = model(mfcc).numpy()[0]
            print(result)
            # 这个result存了每种声音的概率，转换成list，缓存起来
            self.result_buffer[self.result_p % 10] = result
            self.result_p += 1

            # 求每种声音概率平均值，保留最大的
            max_value = 0
            max_id = -1
            # for i in range(2):
            value = self.result_buffer[:, 1].mean()
            if value > max_value:
                max_value = value
                max_id = i

            print(max_value)
            if max_value > 0.8: # set this rate higher so it doesn't buzz as much
                self.recv_sound["id"] = max_id
                self.recv_sound["rate"] = max_value
                self.recv_sound["strength"] = max(ints)  # sum(abs(ints))/len(ints)
            else:
                self.recv_sound["id"] = -1
        return True



    def IsActive(self):
        return self.stream_hd.is_active()

    def Exit(self):
        self.bNeedExit = True


record_hds = [XRecord(1)]

result_str='0,0,0,0'
def send_result():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto("connected".encode("utf8"), ("192.168.4.1", 1234))
    while True:
        sock.sendto(result_str.encode("utf8"), ("192.168.4.1", 1234))


import threading
str_locker=threading.Lock()
threading.Thread(target=send_result).start()

while True:

    finalints = 0
    finalid = ""
    finalmic = -1
    finalconfidence = 0
    result={
        0:0,
        1:0,
        2:0,
        3:0
    }
    for i in range(1):
        while record_hds[i].callback2():
            pass
        if (record_hds[i].recv_sound["id"] != -1) and (record_hds[i].recv_sound["strength"] > finalints):
            finalints = record_hds[i].recv_sound["strength"]
            finalid = record_hds[i].recv_sound["id"]
            finalmic = i
            finalconfidence =record_hds[i].recv_sound["rate"]
            result[i]=1
        else:
            result[i]=0

    str_locker.acquire()
    result_str="%d,%d,%d,%d"%(result[0], result[1], result[2], result[3])
    str_locker.release()
    time.sleep(0.1)
