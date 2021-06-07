import argparse
import os
import mir_eval
import json
import librosa
import librosa.display
import numpy
import util
import yaml
from unmix.source.api import prediction
from helperutils.boolean_argparse import str2bool


def load_audio(file, sample_rate, mono):
    audio, _ = librosa.load(file, sr=sample_rate, mono=mono)
    real_mono = not isinstance(audio[0], numpy.ndarray)
    if real_mono:
        return [audio], audio.shape[0]
    else:
        return [audio[0], audio[1]], audio[0].shape[0]


def generate_stft(audio, fft_length):
    return [librosa.stft(a, fft_length) for a in audio]


parser = argparse.ArgumentParser(description='evaluate a training')
parser.add_argument('--training-folders', nargs='*', required=True,
                    help='path to the folder containing training information')
parser.add_argument('--validation-folder', required=True, help='path to the folder containing validation tracks')
parser.add_argument('--remove-panning', type=str2bool, default=False,
                    help='whether to remove panning from validation tracks')
parser.add_argument('--target-folder',
                    help='where to move all evaluation data, stays in training folder if not defined')
args = parser.parse_args()

for training_folder in args.training_folders:
    util.plot_training(training_folder)

    configuration_file = os.path.join(training_folder, 'configuration.jsonc')
    with open(configuration_file, 'r') as file:
        configuration = json.loads(file.read())

    mono = not configuration['collection']['stereo'] or args.remove_panning
    fft_length = configuration['spectrogram_generation']['fft_length']
    sample_rate = configuration['collection']['sample_rate']

    engine = prediction.create_engine(training_folder)

    evaluations = []

    for track in os.listdir(args.validation_folder):
        track_folder = os.path.join(args.validation_folder, track)
        instrument_stem = None
        rest_stem = None
        for stem in os.listdir(track_folder):
            stem_file = os.path.join(track_folder, stem)
            if os.path.basename(stem).startswith('instrument'):
                instrument_stem = stem_file
            if os.path.basename(stem).startswith('rest'):
                rest_stem = stem_file

        if instrument_stem is not None and rest_stem is not None:
            instrument_sound, instrument_length = load_audio(instrument_stem, sample_rate, mono)
            rest_sound, rest_length = load_audio(rest_stem, sample_rate, mono)
            if instrument_length > rest_length:
                instrument_sound = [s[:rest_length] for s in instrument_sound]
            elif rest_length > instrument_length:
                rest_sound = [s[:instrument_length] for s in rest_sound]
            instrument_stft = generate_stft(instrument_sound, fft_length)
            rest_stft = generate_stft(rest_sound, fft_length)
            mix_stft = []
            for parts in zip(instrument_stft, rest_stft):
                mix_stft.append(parts[0] + parts[1])
            predicted_instrument, predicted_rest = prediction.run_prediction(mix_stft, engine, not mono)

            for sources in zip(instrument_stft, rest_stft, predicted_instrument, predicted_rest):
                (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(
                    numpy.array([librosa.istft(sources[0]), librosa.istft(sources[1])]),
                    numpy.array([sources[2], sources[3]]),
                    compute_permutation=False
                )
                evaluations.append({
                    'track': track,
                    'instrument_sdr': sdr[0].item(),
                    'instrument_sir': sir[0].item(),
                    'instrument_sar': sar[0].item(),
                    'rest_sdr': sdr[1].item(),
                    'rest_sir': sir[1].item(),
                    'rest_sar': sar[1].item()
                })

            util.plot_stems(training_folder, track, mix_stft, instrument_stft, predicted_instrument, fft_length)

    evaluation_file = os.path.join(training_folder, 'evaluation.yaml')
    with open(evaluation_file, 'w') as file:
        yaml.dump(evaluations, file)

    util.print_evaluations(training_folder, configuration, output_file='evaluation.txt')

    if args.target_folder is not None:
        util.copy_evaluation(training_folder, args.target_folder)
