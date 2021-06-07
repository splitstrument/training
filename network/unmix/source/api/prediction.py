from unmix.source.prediction.fileprediction import MixPrediction
from unmix.source.engine import Engine
from unmix.source.configuration import Configuration

import librosa
import numpy
import os
import tensorflow as tf


def inverse_stft(prediction, stereo):
    if stereo:
        return [librosa.istft(channel) for channel in prediction]
    else:
        return [librosa.istft(prediction[0])]


def create_engine(working_directory):
    configuration_file = os.path.join(working_directory, 'configuration.jsonc')
    Configuration.initialize(configuration_file, working_directory, False, True)

    tf.compat.v1.disable_eager_execution()
    engine = Engine()
    path = os.path.join(
        working_directory,
        Configuration.get_path('environment.weights.folder', optional=False),
        Configuration.get('environment.weights.file', optional=False)
    )
    engine.load_weights(path)
    return engine


def run_prediction(mix, engine, remove_panning=False):
    stereo = Configuration.get('collection.stereo', optional=False)
    prediction = MixPrediction(
        engine,
        sample_rate=Configuration.get('collection.sample_rate', optional=False),
        fft_window=Configuration.get('spectrogram_generation.fft_length', optional=False),
        stereo=stereo
    )
    prediction.run(mix, remove_panning=remove_panning)
    return inverse_stft(prediction.instrument, stereo), inverse_stft(prediction.rest, stereo)
