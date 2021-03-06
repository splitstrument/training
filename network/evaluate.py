#!/usr/bin/env python3
# coding: utf8

"""
Measures the accuracy of a training run.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import argparse
import os
import time

from unmix.source.engine import Engine
from unmix.source.helpers import filehelper
from unmix.source.metrics.accuracy import Accuracy
from unmix.source.configuration import Configuration
from unmix.source.logging.logger import Logger
from unmix.source.data.dataloader import DataLoader
from helperutils.boolean_argparse import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--run_folder', default='',
                        help="General training input folder.")
    parser.add_argument('--collection', default='',
                        help="Input folder containing audio files to split vocals and instrumental.")
    parser.add_argument('--test_data_count', default=50,
                        type=int, help="Number of songs to calculate accuracy from.")
    parser.add_argument('--remove_panning', default='False',
                        type=str2bool,
                        help="If panning of stereo input files should be removed by preprocessing.")

    args = parser.parse_args()
    Logger.info("Arguments: ", str(args))
    start = time.time()

    workingdir = filehelper.build_abspath(args.run_folder, os.getcwd())
    configuration = os.path.join(workingdir, 'configuration.jsonc')
    weights = filehelper.get_latest(os.path.join(workingdir, 'weights'), '*weights*.h5')

    Configuration.initialize(configuration, workingdir, False)
    Logger.initialize(False)

    engine = Engine()
    engine.load_weights(weights)

    name = os.path.basename(args.collection)

    training_songs, validation_songs, test_songs = DataLoader.load(args.collection, args.test_data_count)
    engine.accuracy = Accuracy(engine, name)
    engine.test_songs = test_songs
    Logger.info("Found %d songs to measure accuracy." % len(engine.test_songs))

    engine.accuracy.evaluate("measure", remove_panning=args.remove_panning)

    end = time.time()
    Logger.info("Finished processing in %d [s]." % (end - start))
