from copy import deepcopy

import numpy as np
import scipy.fftpack as fft

from recognition.preprocess.wav import *


def mfcc(wav_path, pre_emphasis_coefficient=0.95, frame_time_length_sec=0.025, frame_time_step_sec=0.01,
         hamming_k=0.46, fft_size=None, bottom_freq_hz=0, top_freq_hz=None, filter_banks_count=40,
         coefficients_count=13):
    """
    Вычисляет коэффициенты MFCC по заданному wav-файлу.

    Аргументы:
        wav_path: Путь к wav-файлу.
        pre_emphasis_coefficient: Коэффициент для preemphasis. По-умолчанию 0,95.
        frame_time_length_sec: Длина каждого фрейма в секундах. По-умолчанию 0,025.
        frame_time_step_sec: Шаг каждого фрейма в секундах. По-умолчанию 0,01.
        hamming_k: Коэффициент фильтра Хэмминга. По-умолчанию 0,46.
        fft_size: Размер fft. По умолчанию используется размер фрейма.
        bottom_freq_hz: Гц нижней полосы. По-умолчанию 0.
        top_freq_hz: Гц верхней полосы. По умолчанию 0.
        filter_banks_count: Количество банков фильтров, используемых для вычисления MFCC. По-умолчанию 40.
        coefficients_count: Количество коэффициентов mfcc в выводе. По-умолчанию 13.

    Возвращаемое значение:
        Список коэффициентов MFCC.

    Исключения:
        ValueError: Если путь до WAV файла не является корректным.
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
    Извлекает амплитуды аудиосигнала и частоту дискретизации из заданного WAV-файла.

    Аргументы:
        wav_path: Путь к аудиофайлу формата WAV.

    Возвращаемое значение:
        Кортеж, содержащий:
            - Список целых чисел, представляющих амплитуды сигнала.
            - Частота дискретизации аудиосигнала.

    Исключения:
        ValueError: Если путь до WAV файла не является корректным.
    """
    return extract(wav_path)# wav.extract(wav_path)


def pre_emphases(amplitudes, k=0.95):
    """
    Увеличивает количество энергии на высоких частотах.

    Аргументы:
        amplitudes: Исходные данные, к которым применяется фильтр preemphasis.
        k: Коэффициент preemphasis. От 0 до 1. По-умолчанию 0,95.

    Возвращаемое значение:
        Данные, к которым применен фильтр preemphasis.
    """
    emphased = np.zeros(len(amplitudes), np.float64)
    emphased[0] = amplitudes[0]
    for i in range(1, len(amplitudes)):
        emphased[i] = amplitudes[i] - amplitudes[i - 1] * k

    return emphased


def windowing(emphased, samplerate, frame_time_length_sec=0.025, frame_time_step_sec=0.01, hamming_k=0.46):
    """
    Разбивает сигнал на короткие временные фреймы и применяет окно Хэмминга к каждому кадру.
    Каждый кадр имеет одинаковую длину. Последние кадры дополняются нулями, если необходимо.

    Аргументы:
        emphased: Сигнал для разделения на фреймы.
        samplerate: Частота дискретизации аудиофайла.
        frame_time_length_sec: Длина каждого фрейма в секундах. По-умолчанию 0,025.
        frame_time_step_sec: Шаг каждого фрейма в секундах. По умолчанию 0,01.
        hamming_k: Коэффициент окна Хэмминга. По-умолчанию 0,46.

    Возвращаемое значение:
        Фреймы, на которые был разделен аудиофайл с применением окна Хэмминга к каждому фрейму.
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
    Вычисляет спектр мощности сигнала.

    Аргументы:
        frames: Фреймы, к которым применяется спектр мощности.
        fft_sz: Размер FFT. По-умолчанию равен размеру фрейма.

    Возвращаемое значение:
        Спектр мощности сигнала.
    """
    freq_spectrum = np.absolute(np.fft.rfft(frames, fft_sz))
    return 1.0 / fft_sz * np.square(freq_spectrum)


def get_mel_filterbank(fft_sz, samplerate, filter_banks_count=40, bottom_freq_hz=0, top_freq_hz=None):
    """
    Генерирует банк треугольных мел-фильтров для преобразования частотного спектра в мел-шкалу.

    Аргументы:
        fft_sz: Размер FFT.
        samplerate: Частота дискретизации входного сигнала.
        filter_banks_count: Количество мел-фильтров. По-умолчанию: 40.
        bottom_freq_hz: Минимальная частота в Гц. По-умолчанию: 0.
        top_freq_hz: Максимальная частота в Гц. По-умолчанию: samplerate // 2.

    Возвращаемое значение:
        Numpy массив мел-фильтров.
    """
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
    """
    Преобразует частоту из герц (Hz) в меловую шкалу (Mel).

    Аргументы:
        hz: Частота в герцах.

    Возвращаемое значение:
        Частота в меловой шкале.
    """
    return 2595 * np.log10(1 + hz / 700)


def mel2hz(mel):
    """
    Преобразование из меловой шкалы (Mel) в частоту в герцах (Hz).

    Аргументы:
        mel: Частота в Mel.

    Возвращаемое значение:
        Частота в герцах.
    """
    return 700 * (10 ** (mel / 2595) - 1)


def log_mel_spectrum(powspectrum, samplerate, fft_sz, filter_banks_count=40, bottom_freq_hz=0, top_freq_hz=None):
    """
    Вычисляет логарифмический мел-спектр из энергетического спектра.

    Аргументы:
        powspectrum: Энергетический спектр.
        samplerate: Частота дискретизации входного сигнала.
        fft_sz: Размер FFT.
        filter_banks_count: Количество мел-фильтров. По-умолчанию: 40.
        bottom_freq_hz: Минимальная частота в Гц. По-умолчанию: 0.
        top_freq_hz: Максимальная частота в Гц. По-умолчанию: samplerate // 2.

    Возвращаемое значение:
        Логарифмический мел-спектр.
    """
    fbanks = get_mel_filterbank(fft_sz, samplerate, filter_banks_count, bottom_freq_hz, top_freq_hz)
    melspectrum = np.dot(powspectrum, fbanks.T)
    return np.log(np.where(melspectrum == 0, np.finfo(float).eps, melspectrum))


def dct(melspectrum, coefficients_count=13):
    """
    Вычисляет дискретное косинусное преобразование.
    Используется для преобразования логарифмического
    мел-спектра в MFCC.

    Аргументы:
        melspectrum: Логарифмический мел-спектр.
        coefficients_count: Возвращаемое количество коэффициентов MFCC.

    Возвращаемое значение:
        Коэффициенты MFCC в количестве coefficients_count.
    """
    return fft.dct(melspectrum, type=2, axis=1, norm='ortho')[:, 1: coefficients_count + 1]


if __name__ == '__main__':
    path = "C:/Users/PeterA/Desktop/vkr/one.wav"#"C:/Users/PeterA/Desktop/yabba/one.wav"
    mfccs = mfcc(wav_path=path)
    print(mfccs)

    # sample_rate, amplitude = mfcc.adc()
    # pre_emphase = mfcc.pre_emphases(amplitude)
    # frames = mfcc.windowing(pre_emphase, sample_rate)
    # dft_spectrum, nfft = mfcc.dft_power_spectrum(frames)
    # mel_power_spectrum = mfcc.mel_power_spectrum(dft_spectrum, nfft, sample_rate)
    # mfccs = mfcc.features(mel_power_spectrum)
    # # print(len(mfccs))
    # print(len(mfccs[0]))