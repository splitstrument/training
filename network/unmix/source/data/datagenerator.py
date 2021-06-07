#!/usr/bin/env python3
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

import keras
import numpy as np

from unmix.source.configuration import Configuration
from unmix.source.data.batchitem import BatchItem
from unmix.source.data.song import Song
from unmix.source.logging.logger import Logger


def limit_items_of_song(items, limit):
    fft_scaling_factor = 1536 / Configuration.get('spectrogram_generation.fft_length', default=1536)
    limit *= fft_scaling_factor
    middle = len(items) / 2
    limit_half = limit / 2
    start = int(max(0, middle - limit_half))
    end = int(min(len(items), start + limit))
    return items[start:end]


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    index = None

    def __init__(self, name, engine, collection, transformer, accuracy=None):
        self.name = name
        self.collection = collection
        self.transformer = transformer
        self.batch_size = Configuration.get('training.batch_size', default=8)
        self.epoch_shuffle = Configuration.get('training.epoch.shuffle')
        self.engine = engine
        self.accuracy = accuracy
        self.count = 0
        self.on_epoch_end()

    def generate_index(self):
        self.index = np.array([])
        for file in self.collection:
            try:
                song = Song(file)
                items = [BatchItem(file, i, song.name) for i in range(self.transformer.calculate_items(song.width))]

                if self.transformer.shuffle:
                    np.random.shuffle(items)

                # limit number of items per song if configured
                limit_items_per_song = Configuration.get('training.limit_items_per_song', default=0)
                if limit_items_per_song > 0:
                    items = limit_items_of_song(items, limit_items_per_song)

                self.index = np.append(self.index, items)
            except Exception as e:
                if self.count == 0:
                    Logger.warn("Skip file while generating index: %s" % str(e.args))

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.index) // self.batch_size

    def __getitem__(self, i):
        """Generate one batch of data"""
        subset = self.index[i * self.batch_size:(i + 1) * self.batch_size]
        x, y = self.__data_generation(subset)
        return x, y

    def on_epoch_end(self):
        """Updates index after each epoch"""
        Logger.debug("%s epoch %d ended." % (self.name, self.count))
        self.generate_index()
        if self.epoch_shuffle:
            np.random.shuffle(self.index)
        test_frequency = Configuration.get('collection.test_frequency', default=0)
        if self.engine.test_songs and self.accuracy and \
                test_frequency > 0 and self.count % test_frequency == 0:
            self.accuracy.evaluate(self.count)
        self.count += 1

    def __data_generation(self, subset):
        """Generates data containing batch_size samples"""
        x = []
        y = []

        for item in subset:
            if item is not None:
                rest, instrument = item.load()
                rest, instrument = self.transformer.run(item.name, rest, instrument, item.index)
                x.append(rest)
                y.append(instrument)

        return np.array(x), np.array(y)
