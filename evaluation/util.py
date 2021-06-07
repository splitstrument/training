import statistics
import yaml
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import numpy
import librosa
import librosa.display
import sys
from tabulate import tabulate


def print_evaluations(training_folder, configuration, output_file=None):
    evaluation_file = os.path.join(training_folder, 'evaluation.yaml')
    with open(evaluation_file, 'r') as file:
        evaluations = yaml.load(file, Loader=yaml.FullLoader)

    trainable_params = None
    log_file = os.path.join(training_folder, 'logs.txt')
    with open(log_file, 'r') as file:
        search = re.search(r'Trainable params: ([0-9,]+)', file.read())
        if search is not None:
            groups = search.groups()
            if len(groups) >= 1:
                trainable_params = groups[0]

    instrument_sdr = []
    instrument_sir = []
    instrument_sar = []
    rest_sdr = []
    rest_sir = []
    rest_sar = []

    for evaluation in evaluations:
        instrument_sdr.append(evaluation['instrument_sdr'])
        instrument_sir.append(evaluation['instrument_sir'])
        instrument_sar.append(evaluation['instrument_sar'])
        rest_sdr.append(evaluation['rest_sdr'])
        rest_sir.append(evaluation['rest_sir'])
        rest_sar.append(evaluation['rest_sar'])

    median_instrument_sdr = statistics.median(instrument_sdr)
    median_instrument_sir = statistics.median(instrument_sir)
    median_instrument_sar = statistics.median(instrument_sar)
    median_rest_sdr = statistics.median(rest_sdr)
    median_rest_sir = statistics.median(rest_sir)
    median_rest_sar = statistics.median(rest_sar)

    stereo = configuration['collection']['stereo']
    fft_length = configuration['spectrogram_generation']['fft_length']
    sample_rate = configuration['collection']['sample_rate']
    collection_name = configuration['collection'].get('name')

    folders = configuration['collection'].get('folders', [])
    if len(folders) > 0:
        folder = ', '.join([f['path'] for f in folders])
    else:
        folder = configuration['collection']['folder']

    dataset_infos = []
    if len(folders) > 0:
        for f in folders:
            dataset_infos.append('Dataset %s with ratio %s' % (f['path'], f['ratio']))

    song_limit = configuration['collection']['song_limit']
    item_limit = configuration['training']['limit_items_per_song']
    model_name = configuration['training']['model']['name']
    additional_info = []
    if model_name == 'Hourglass':
        additional_info.append('Stacks: %s' % configuration['training']['model']['options']['stacks'])
        additional_info.append('Filters: %s' % configuration['training']['model']['options']['hg_module']['filters'])
    batch_size = configuration['training']['batch_size']
    loss_function = configuration['training']['loss_function']
    optimizer = configuration['training']['optimizer']['name']
    transformation = configuration['transformation']['name']

    default_output = None
    if output_file is not None:
        default_output = sys.stdout
        output_file_path = os.path.join(training_folder, output_file)
        if os.path.isfile(output_file_path):
            os.remove(output_file_path)
        os.mknod(output_file_path)
        sys.stdout = open(output_file_path, 'w')

    print('%s / %s' % (collection_name, folder) if collection_name is not None else 'Dataset %s' % folder)
    print('%s / %s Hz / %s FFT window / %s' % (model_name, sample_rate, fft_length, 'Stereo' if stereo else 'Mono'))
    if trainable_params is not None:
        print('Trainable params: %s' % trainable_params)
    if song_limit > 1:
        used_songs = str(song_limit)
    elif song_limit > 0:
        used_songs = str(song_limit * 100) + '% of '
    else:
        used_songs = 'All'
    used_items = int(item_limit * 1536 / fft_length) if item_limit > 0 else 'all'
    print('%s songs with %s items each and %s per batch' % (used_songs, used_items, batch_size))
    print('Loss function %s, optimizer %s, transformation %s' % (loss_function, optimizer, transformation))
    print()
    if len(dataset_infos) > 0:
        print('Information about dataset')
        for dataset_info in dataset_infos:
            print(dataset_info)
        print()
    if len(additional_info) > 0:
        print('Additional information for %s model' % model_name)
        for info in additional_info:
            print(info)
        print()
    data = [
        ['Instrument', median_instrument_sdr, median_instrument_sir, median_instrument_sar],
        ['Rest', median_rest_sdr, median_rest_sir, median_rest_sar],
    ]
    print(tabulate(data, headers=["", "SDR", "SIR", "SAR"]))
    print()

    if output_file is not None:
        sys.stdout.close()
        sys.stdout = default_output


def plot_training(folder, save_figure=True):
    csv_filename = 'results.csv'
    filepath = os.path.join(folder, csv_filename)
    with open(filepath) as file:
        if len(file.readlines()) <= 0:
            return

    epoch_data = pd.read_csv(filepath, index_col=0, sep=';', usecols=['epoch', 'loss', 'val_loss'])
    epoch_data.plot()
    if save_figure:
        plt.savefig(os.path.join(folder, 'plots', 'results_plot'))
    else:
        plt.show()

    plt.close()


def copy_evaluation(training_folder, target_folder):
    folder_name = os.path.basename(os.path.normpath(training_folder))
    new_folder = os.path.join(target_folder, folder_name)
    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)

    configuration_file = os.path.join(training_folder, 'configuration.jsonc')
    shutil.copy(configuration_file, new_folder)
    evaluation_file = os.path.join(training_folder, 'evaluation.yaml')
    shutil.copy(evaluation_file, new_folder)
    evaluation_text = os.path.join(training_folder, 'evaluation.txt')
    shutil.copy(evaluation_text, new_folder)
    evaluation_file = os.path.join(training_folder, 'configuration.jsonc')
    shutil.copy(evaluation_file, new_folder)
    log_file = os.path.join(training_folder, 'logs.txt')
    shutil.copy(log_file, new_folder)
    results_file = os.path.join(training_folder, 'results.csv')
    shutil.copy(results_file, new_folder)
    plot_folder = os.path.join(training_folder, 'plots')
    shutil.copytree(plot_folder, os.path.join(new_folder, 'plots'), dirs_exist_ok=True)


def save_spectrogram_image(plot_folder, stfts, labels, name):
    fig, axs = plt.subplots(nrows=len(stfts), sharex='col')
    for label, stft, ax in zip(labels, stfts, axs):
        img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=numpy.max), y_axis='log', x_axis='time', ax=ax)
        ax.set_title(label)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.set_figheight(8)

    fig.tight_layout()
    plt.savefig(os.path.join(plot_folder, name), format='svg')
    plt.close()


def plot_stems(training_folder, track, mix_stft, instrument_stft, predicted_instrument, fft_length):
    plot_folder = os.path.join(training_folder, 'plots')
    labels = ['Mix', 'Instrument', 'Prediction']
    for index, sources in enumerate(zip(mix_stft, instrument_stft, predicted_instrument)):
        sources = (sources[0], sources[1], librosa.stft(sources[2], fft_length))
        save_spectrogram_image(plot_folder, sources, labels, '%s_spectrograms_%s.svg' % (track, index))
