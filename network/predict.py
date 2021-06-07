#!/usr/bin/env python3
# coding: utf8

"""
Predicts vocal and/or instrumental for a song.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import argparse
import glob
import os
import time
import tensorflow as tf

from unmix.source.engine import Engine
from unmix.source.helpers import filehelper
from helperutils.boolean_argparse import str2bool
from helperutils.audio_file_checker import is_accepted_audio_file
from unmix.source.prediction.fileprediction import FilePrediction
from unmix.source.prediction.youtubeprediction import YoutTubePrediction
from unmix.source.configuration import Configuration
from unmix.source.logging.logger import Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executes a training session.")
    parser.add_argument('--run_folder', default='', help="General training input folder (overwrites other parameters).")
    parser.add_argument('--configuration', default='', help="Environment and training configuration.")
    parser.add_argument('--weights', default='', help="Pretrained weights file (overwrites configuration).")
    parser.add_argument('--workingdir', default=os.getcwd(), help="Working directory (default: current directory).")
    parser.add_argument('--sample_rate', default=44100, type=int,
                        help="Target sample rate which the model can process.")
    parser.add_argument('--fft_window', default=1536, type=int, help="FFT window size the model was trained on.")
    parser.add_argument('--remove_panning', default='False', type=str2bool,
                        help="If panning of stereo input files should be removed by preprocessing.")
    parser.add_argument('--song', default='', help="Input audio file to split some instrument off the rest.")
    parser.add_argument('--songs', default='./temp/songs',
                        help="Input folder containing audio files to split some instrument off the rest.")
    parser.add_argument('--youtube', default='', type=str, help="Audio from a youtube video as file (or later stream).")

    tf.compat.v1.disable_eager_execution()

    args = parser.parse_args()
    start = time.time()

    if args.run_folder:
        args.workingdir = filehelper.build_abspath(
            args.run_folder, os.getcwd())
        args.configuration = os.path.join(
            args.workingdir, 'configuration.jsonc')
        args.weights = filehelper.get_latest(os.path.join(
            args.workingdir, 'weights'), '*weights*.h5')

    Configuration.initialize(args.configuration, args.workingdir, False, True)
    Logger.initialize(False)

    if args.run_folder:
        prediction_folder = Configuration.build_path('predictions')
        args.sample_rate = Configuration.get("collection.sample_rate")
    else:
        prediction_folder = ''

    if os.path.isdir(args.song):
        args.songs = args.song
        args.song = ''

    Logger.info("Arguments: ", str(args))

    song_files = []
    if args.song:
        song_files = [args.song]
    if args.songs:
        for file in glob.iglob(filehelper.build_abspath(args.songs, os.getcwd()) + '**/*', recursive=True):
            is_audio_file, _ = is_accepted_audio_file(file)
            if is_audio_file:
                song_files.append(file)

    Logger.info("Found %d songs to predict." % len(song_files))

    engine = Engine()
    engine.load_weights(args.weights)

    stereo = Configuration.get("collection.stereo", default=False)
    fft_window = Configuration.get('spectrogram_generation.fft_length', default=args.fft_window)
    sample_rate = Configuration.get('collection.sample_rate', default=args.sample_rate)

    for song_file in song_files:
        try:
            prediction = FilePrediction(engine, sample_rate=sample_rate, fft_window=fft_window, stereo=stereo)
            prediction.run(song_file, remove_panning=args.remove_panning)
            prediction.save(song_file, prediction_folder)
        except Exception as e:
            Logger.error("Error while predicting song '%s': %s." % (song_file, str(e)))

    if args.youtube:
        prediction = YoutTubePrediction(engine, sample_rate=sample_rate, fft_window=fft_window, stereo=stereo)
        path, name, _ = prediction.run(args.youtube)
        prediction.save(name, path)

    end = time.time()
    Logger.info("Finished processing in %d [s]." % (end - start))
