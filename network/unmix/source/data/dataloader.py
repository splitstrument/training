#!/usr/bin/env python3
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import hashlib
import glob
import os
import random

from unmix.source.configuration import Configuration
from unmix.source.data.song import Song
from unmix.source.logging.logger import Logger


class DataLoader(object):

    @staticmethod
    def load(path=None, test_data_count=None):
        if path is None:
            folders = Configuration.get('collection.folders')
            if folders is None:
                return DataLoader.loadDataset(Configuration.get('collection.folder', optional=False), test_data_count)
            else:
                return DataLoader.loadMultipleDatasets(folders, test_data_count)

        else:
            return DataLoader.loadDataset(path, test_data_count)

    @staticmethod
    def loadDataset(path, test_data_count):
        files = DataLoader.loadFiles(path)

        training_files, validation_files, test_files = DataLoader.splitDataset(files, test_data_count)

        Logger.debug(
            "Found %d songs for training and %d songs for validation." % (len(training_files), len(validation_files)))
        if test_files is not None:
            test_frequency = Configuration.get('collection.test_frequency', default=0)
            Logger.debug("Use %d songs for tests after every %d epoch." % (len(test_files), test_frequency))

        if len(training_files) == 0:
            Logger.warn("No training files assigned.")
        if len(validation_files) == 0:
            Logger.warn("No validation files assigned.")
        return training_files, validation_files, test_files

    @staticmethod
    def loadFiles(path, ignore_song_limit=False):
        if path is None:
            path = Configuration.get_path('collection.folder', False)
        instrument_filter = os.path.join(path, '**', '%s*.wav' % Song.PREFIX_INSTRUMENT)
        files_instrument = [os.path.dirname(file) for file in glob.iglob(instrument_filter, recursive=True)]
        rest_filter = os.path.join(path, '**', '%s*.wav' % Song.PREFIX_REST)
        files_rest = [os.path.dirname(file) for file in glob.iglob(rest_filter, recursive=True)]

        files = [f for f in files_instrument if f in files_rest]  # make sure both instrument and rest file exists
        skipped_count = len(set(files_instrument) - set(files_rest)) + len(set(files_rest) - set(files_instrument))
        Logger.debug(f"Skipped {skipped_count} files (incomplete instrument/rest pair)")

        # Sort files by hash value of folder to guarantee a consistent order
        files.sort(key=lambda x: hashlib.md5(os.path.basename(x).encode('utf-8', 'surrogatepass')).hexdigest())

        song_limit = Configuration.get('collection.song_limit', default=0)
        if not ignore_song_limit and song_limit > 0:
            if song_limit <= 1:  # Configuration as percentage share
                song_limit = song_limit * len(files)
            song_limit = min(int(song_limit), len(files))
            files = files[:song_limit]

        return files

    @staticmethod
    def splitDataset(files, test_data_count):
        test_files = None
        test_frequency = Configuration.get('collection.test_frequency', default=0)
        if not test_data_count:
            test_data_count = Configuration.get('collection.test_data_count', default=0)
        if test_data_count > 0:
            test_data_count = int(test_data_count)
            test_files = files[-test_data_count:]
            files = files[:len(files) - test_data_count]

        validation_ratio = Configuration.get('collection.validation_ratio', default=0.2)
        validation_files = files[:int(len(files) * validation_ratio)]
        training_files = files[len(validation_files):]

        return training_files, validation_files, test_files

    @staticmethod
    def loadMultipleDatasets(folders, test_data_count):
        datasets = []
        ratio_sum = 0
        smallest_dataset_length = None
        smallest_dataset_ratio = None
        for folder in folders:
            ratio = folder['ratio']
            dataset = DataLoader.loadFiles(folder['path'], True)
            datasets.append((dataset, ratio, folder['path']))
            ratio_sum = ratio_sum + ratio
            dataset_length = len(dataset)
            if smallest_dataset_length is None or dataset_length < smallest_dataset_length:
                smallest_dataset_length = dataset_length
                smallest_dataset_ratio = ratio

        target_song_count = ratio_sum / smallest_dataset_ratio * smallest_dataset_length
        song_limit = Configuration.get('collection.song_limit', default=0)
        if song_limit < target_song_count:
            if song_limit >= 1:
                target_song_count = song_limit
            elif song_limit > 0:
                target_song_count = target_song_count * song_limit

        training_files = []
        validation_files = []
        test_files = []
        for dataset, ratio, folder in datasets:
            requested_file_count = int(ratio / ratio_sum * target_song_count)
            files = dataset[:requested_file_count]
            print('Loaded %s files from %s' % (len(files), folder))
            training, validation, test = DataLoader.splitDataset(files, test_data_count)
            training_files.extend(training)
            validation_files.extend(validation)
            if test is not None:
                test_files.extend(test)

        return training_files, validation_files, test_files
