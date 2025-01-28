import os

class WavProcessor:

    def _unpack(f, c):
        bytes_arr = [int.from_bytes(f.read(1)) for _ in range(c)]
        val = 0
        for i in reversed(range(c)):
            val += bytes_arr[i] * 256 ** i
        return val


    def extract(wav_path, channel=1):
        with open(wav_path, 'rb') as f:
            if f.read(4) != b"RIFF":
                raise ValueError("Invalid file format: Expected 'RIFF'")
            f.seek(18, os.SEEK_CUR)
            channels = unpack(f, 2)  # количество каналов
            if channel > channels or channel < 1:
                raise ValueError(f"Wrong channel. In audio: {channels}. Requested {channel} channel!")
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
