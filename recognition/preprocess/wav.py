import os


def unpack(f, c):
    """
    Считывает заданное количество байт в массив.
    Преобразует массив байт в число.

    Аргументы:
        f: Файловый объект, открытый в бинарном режиме 'rb'.
        c: Количество байт, которые требуется считать.

    Возвращаемое значение:
        Целое число, полученное из считанного массива байт.
    """
    bytes_arr = [int.from_bytes(f.read(1)) for _ in range(c)]
    val = 0
    for i in reversed(range(c)):
        val += bytes_arr[i] * 256 ** i
    return val


def extract(wav_path, channel=1):
    """
    Обрабатывает WAV-файл и извлекает из него данные
    об амплитуде от времени для указанного канала,
    а также частоту дискретизации.

    Аргументы:
        wav_path: Путь к WAV-файлу.
        channel: Индекс указанного канала (начиная с 1) для
            многоканальных файлов. По-умолчанию: 1.

    Возвращаемое значение:
        Кортеж, содержащий:
            - список амплитуд от времени
            - частоту дискретизации WAV-файла

    Исключения:
        ValueError: Если по указанному пути находится не WAV-файл,
            либо если указан несуществующий номер канала.
        FileNotFoundError: Если по указанному пути файл отсутствует.
    """
    with open(wav_path, 'rb') as f:
        if f.read(4) != b"RIFF":
            raise ValueError("Некорректный формат файла: Ожидалось встретить 'RIFF'")
        f.seek(18, os.SEEK_CUR)
        channels = unpack(f, 2)  # количество каналов
        if channel > channels or channel < 1:
            raise ValueError(f"Некорректный канал. В аудио: {channels}. Запрашивается {channel} канал!")
        sample_rate = unpack(f, 4)  # частота дискретизации
        f.seek(6, os.SEEK_CUR)
        bps = unpack(f, 2) // 8  # байтов в сэмпле
        f.seek(42 + (channel - 1) * bps, os.SEEK_CUR)
        data = []
        while True:
            item = f.read(bps)
            if item == b'':
                break
            data.append(int.from_bytes(item, byteorder='little', signed=True))
            f.seek(bps * (channels - 1), os.SEEK_CUR)
        return data, sample_rate