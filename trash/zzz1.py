import numpy as np
import scipy
from hmmlearn import hmm

if __name__ == "__main__":


    m = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print(np.linalg.det(m))

    fake_inv = scipy.linalg.inv(m) #np.linalg.pinv(m)
    print(fake_inv)

    hmm.GMMHMM.
    # print(np.dot(m, fake_inv))

    exit(0)

    t1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    t2 = [0, 5e100, 1e100, 5e100, 0, 5e100, 1e100, 5e100, 0, 5e100, 1e100, 5e100, 0, 5e100, 1e100, 5e100, 0, 5e100, 1e100, 5e100, 0]

    y1 = np.fft.rfft(t1)
    y2 = np.fft.rfft(t2)

    gh = hmm.GMMHMM

    w1 = np.fft.irfft(np.log(y2))
    w2 = np.fft.irfft(np.log(y2) + np.log(y1))
    print(w1)
    print(w2)
    print(w2 - w1)

  #
  #   y3 = np.fft.irfft(np.log(y1))
  #   print(np.fft.irfft(y3))
  #
  #   y4 = np.fft.irfft(np.log(y1)) + np.fft.irfft(np.log(y2))
  #   print(y4)
