from copy import deepcopy

import numpy as np
import scipy.fftpack as fft

import wav


def mfcc(wav_path, pre_emphasis_coefficient=0.95, frame_time_length_sec=0.025, frame_time_step_sec=0.01,
         hamming_k=0.46, fft_size=None, bottom_freq_hz=0, top_freq_hz=None, filter_banks_count=40,
         coefficients_count=13):
    """
    Computes MFCCs coefficients given the wav file.

    :param wav_path: Path to wav file.
    :param pre_emphasis_coefficient: Coefficient for preemphasis. Defaults to 0.95.
    :param frame_time_length_sec: Length of each frame in seconds. Defaults to 0.025.
    :param frame_time_step_sec: Step of each frame in seconds. Defaults to 0.01.
    :param hamming_k: Hamming window coefficient. Defaults to 0.46
    :param fft_size: Size of fft. Defaults to size of frame.
    :param bottom_freq_hz: Hz of bottom band. Defaults to 0.
    :param top_freq_hz: Hz of top band. Defaults to 0.
    :param filter_banks_count: Number of filter banks used to compute MFCCs. Defaults to 40.
    :param coefficients_count: Number of mfcc coefficients in output. Defaults to 13
    :return: list of mfcc coefficients
    :raises: ValueError: If the WAV path is not a valid WAV file.
    """
    amplitudes, samplerate = adc(wav_path)
    emphased = pre_emphases(amplitudes, pre_emphasis_coefficient)
    frames = windowing(emphased, samplerate, frame_time_length_sec, frame_time_step_sec, hamming_k)
    fft_size = fft_size or len(frames[0])
    powspectrum = power_spectrum(frames, fft_size)
    melspectrum = log_mel_spectrum(powspectrum, samplerate, fft_size, filter_banks_count, bottom_freq_hz, top_freq_hz)
    return dct(melspectrum, coefficients_count)


def adc(wav_path):
    """
    Extracts the amplitudes of the audio signal and the sample rate from a given WAV file.

    :param wav_path: The file path to the WAV audio file.
    :return: A tuple containing:
        - List of integers representing the amplitudes of the signal.
        - The sample rate of audio signal.
    :raises ValueError: If the WAV path is not a valid WAV file.
    """
    return wav.extract(wav_path)


def pre_emphases(amplitudes, k=0.95):
    """
    Boosts the amount of energy in the high frequencies.

    :param amplitudes: The initial data to which apply pre_emphasis filter.
    :param k: The coefficient of pre_emphasis. From 0 to 1. Defaults to 0.95.
    :return: Data to which pre_emphasis filter is applied.
    """
    emphased = np.zeros(len(amplitudes), np.float64)
    emphased[0] = amplitudes[0]
    for i in range(1, len(amplitudes)):
        emphased[i] = amplitudes[i] - amplitudes[i - 1] * k

    return emphased


def windowing(emphased, samplerate, frame_time_length_sec=0.025, frame_time_step_sec=0.01, hamming_k=0.46):
    """
    Splits signal into short-time frames and applies hamming window to each frame.
    Each frame is same length. Last frames padded with zeros if needed.

    :param emphased: Signal to split into frames.
    :param samplerate: Audio file sampling rate.
    :param frame_time_length_sec: Length of each frame in seconds. Defaults to 0.025.
    :param frame_time_step_sec: Step of each frame in seconds. Defaults to 0.01.
    :param hamming_k: Hamming window coefficient. Defaults to 0.46.
    :return: Frames to which audio file was split with hamming window applied to each frame.
    """
    frame_step = int(frame_time_step_sec * samplerate)
    frame_length = int(frame_time_length_sec * samplerate)

    frames = []
    frame_begin = 0
    size = len(emphased)
    while frame_begin < size:
        frames.append(deepcopy(emphased[frame_begin:min(frame_begin + frame_length, size - 1)]))
        real_length = len(frames[-1])

        for i in range(real_length):
            frames[-1][i] *= 1 - hamming_k - hamming_k * np.cos(
                (2 * np.pi * i) / (frame_length - 1))

        if real_length < frame_length:
            padding = np.zeros(frame_length - real_length)
            frames[-1] = np.append(frames[-1], padding)
            break
        frame_begin += frame_step

    return frames


def power_spectrum(frames, fft_sz=512):
    """
    Computes the power spectrum of a signal.
    :param frames: Frames to which power spectrum applies.
    :param fft_sz: Size of FFT. Defaults to size of frame.
    :return: The power spectrum of the signal.
    """
    freq_spectrum = np.absolute(np.fft.rfft(frames, fft_sz))
    return 1.0 / fft_sz * np.square(freq_spectrum)


def get_mel_filterbank(fft_sz, samplerate, filter_banks_count=40, bottom_freq_hz=0, top_freq_hz=None):
    top_freq_hz = top_freq_hz or samplerate // 2
    bottom_freq_mel, top_freq_mel = hz2mel(bottom_freq_hz), hz2mel(top_freq_hz)

    hzs = mel2hz(np.linspace(bottom_freq_mel, top_freq_mel, filter_banks_count + 2))
    inds = np.floor((fft_sz + 1) * hzs / samplerate).astype(int)

    fbanks = np.zeros((filter_banks_count, fft_sz // 2 + 1))
    for i in range(0, filter_banks_count):
        k = inds[i]
        while k < inds[i + 1]:
            fbanks[i, k] = (k - inds[i]) / (inds[i + 1] - inds[i])
            k += 1
        while k < inds[i + 2]:
            fbanks[i, k] = (inds[i + 2] - k) / (inds[i + 2] - inds[i + 1])
            k += 1

    return fbanks


def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel2hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def log_mel_spectrum(powspectrum, samplerate, fft_sz, filter_banks_count=40, bottom_freq_hz=0, top_freq_hz=None):
    fbanks = get_mel_filterbank(fft_sz, samplerate, filter_banks_count, bottom_freq_hz, top_freq_hz)
    melspectrum = np.dot(powspectrum, fbanks.T)
    return np.log(np.where(melspectrum == 0, np.finfo(float).eps, melspectrum))


def dct(melspectrum, coefficients_count=13):
    # print(fft.dct(melspectrum, type=2, axis=1, norm='ortho')[:, 1: coefficients_count + 1])
    # print("=============")
    # print(fft.ifft(melspectrum)[:, 1: coefficients_count + 1])
    return fft.dct(melspectrum, type=2, axis=1, norm='ortho')[:, 1: coefficients_count + 1]


if __name__ == '__main__':
    path = "C:/Users/PeterA/Desktop/yabba/one.wav"
    mfccs = mfcc()
    print(mfccs)

    # sample_rate, amplitude = mfcc.adc()
    # pre_emphase = mfcc.pre_emphases(amplitude)
    # frames = mfcc.windowing(pre_emphase, sample_rate)
    # dft_spectrum, nfft = mfcc.dft_power_spectrum(frames)
    # mel_power_spectrum = mfcc.mel_power_spectrum(dft_spectrum, nfft, sample_rate)
    # mfccs = mfcc.features(mel_power_spectrum)
    # # print(len(mfccs))
    # print(len(mfccs[0]))
