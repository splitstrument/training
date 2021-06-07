#!/usr/bin/env python3
# coding: utf8

'''
Model of a song to train with.
'''

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = 'info@unmix.io'

import glob
import os

from unmix.source.data.track import Track
from unmix.source.exceptions.dataerror import DataError
from unmix.source.helpers import spectrogramhandler
from unmix.source.data.spectrogram_generator import generate_spectrogram


class Song(object):
    PREFIX_REST = 'rest_'
    PREFIX_INSTRUMENT = 'instrument_'

    def __init__(self, folder):
        instrument_file = None
        for file in glob.iglob(os.path.join(folder, '%s*.wav' % Song.PREFIX_INSTRUMENT)):
            instrument_file = file
            break
        rest_file = None
        for file in glob.iglob(os.path.join(folder, '%s*.wav' % Song.PREFIX_REST)):
            rest_file = file
            break
        if instrument_file is None or rest_file is None:
            raise DataError(folder, 'missing instrument or rest track')
        data_instrument = generate_spectrogram(instrument_file)
        self.folder = folder
        self.height = data_instrument['height']
        data_rest = None
        if rest_file is not None:
            data_rest = generate_spectrogram(rest_file)
            self.width = min(int(data_instrument['width']), int(data_rest['width']))
        self.depth = data_instrument['depth']
        self.fft_window = data_instrument['fft_window']
        self.sample_rate = data_instrument['sample_rate']
        self.collection = data_instrument['collection']
        self.name = data_instrument['song']
        self.instrument = Track('instrument', self.height, self.width,
                                self.depth, instrument_file, data_instrument)
        self.rest = Track('rest', self.height, self.width,
                          self.depth, rest_file, data_rest)
        self.mix = Track('mix', self.height, self.width, self.depth)

    def load(self, remove_panning=False, clean_up=True):
        if not self.mix.initialized and self.rest is not None:
            try:
                # After this step all tracks are initialized
                self.mix.mix(self.instrument, self.rest)
                if remove_panning:
                    self.mix.channels = spectrogramhandler.remove_panning(self.mix.channels)
            except Exception as e:
                raise DataError(self.folder, str(e))
            finally:
                if clean_up:
                    self.clean_up()
        return self.mix.channels, self.instrument.channels

    def clean_up(self):
        self.rest = []
