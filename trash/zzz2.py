import sys

import numpy as np
from python_speech_features import sigproc
from data_preparation import mfcc

np.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    # Загружаем аудиофайл
    path = "C:/Users/PeterA/Desktop/yabba/one.wav"
    # mfcc = MFCC(path)

    adc, sr = mfcc.adc(path)
    prf = mfcc.pre_emphases(adc)
    frame = sigproc.framesig(prf, 0.025 * sr, 0.01 * sr, lambda x: np.hamming(x))

    # frames = mfcc.windowing(prf, sr)
    # print(frame[132] - np.array(frames[132]))

    # xxx = base.get_filterbanks(40, 1200, 48000)
    xxxx = mfcc.get_mel_filterbank(1200, 48000, 40)

    # for i in range(len(xxxx)):
    #     print(xxxx[i].tolist())

    # for i in range(len(xxx)):
    #     print(xxx[i].tolist())

    # for i in range(len(frames)):
    #     for j in range(len(frames[0])):
    #         if math.fabs(frame[i][j] - frames[i][j]) >= 1e-10:
    #             print("OOPS!", frames[i][j], frame[i][j], i, j)
