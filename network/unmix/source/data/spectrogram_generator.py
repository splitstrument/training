import librosa
import os
import numpy as np
from unmix.source.configuration import Configuration


def generate_stft(audio, fft_length):
    stft = librosa.stft(audio, fft_length)
    dimensions = (stft.shape[0], stft.shape[1], 2)
    return dimensions, stft


def generate_spectrogram(file):
    mono = not Configuration.get('collection.stereo', default=False)
    sample_rate = Configuration.get('collection.sample_rate', default=44100)
    fft_length = Configuration.get('spectrogram_generation.fft_length', default=1536)
    audio, sample_rate = librosa.load(file, mono=mono, sr=sample_rate)
    mono = not isinstance(audio[0], np.ndarray)
    if mono:
        dimensions, spectrogram = generate_stft(audio, fft_length)
        spectrograms = [spectrogram]
    else:
        dimensions, spectrogram1 = generate_stft(audio[0], fft_length)
        dimensions, spectrogram2 = generate_stft(audio[1], fft_length)
        spectrograms = [spectrogram1, spectrogram2]
    return {
        'spectrograms': spectrograms,
        'height': dimensions[0],
        'width': dimensions[1],
        'depth': dimensions[2],
        'fft_window': fft_length,
        'sample_rate': sample_rate,
        'mono': mono,
        'song': os.path.basename(os.path.dirname(file)),
        'collection': Configuration.get('collection.folder',
                                        default=os.path.basename(os.path.dirname(os.path.dirname(file))))
    }
