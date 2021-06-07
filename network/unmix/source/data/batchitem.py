#!/usr/bin/env python3
# coding: utf8

"""
Batch item with song and chop offset.
"""

__author__ = 'David Flury, Andreas Kaufmann, Raphael Müller'
__email__ = "info@unmix.io"

from unmix.source.data.song import Song

from cachetools import LRUCache, cached


@cached(cache=LRUCache(maxsize=5))
def load_song(file):
    return Song(file).load()


class BatchItem(object):

    def __init__(self, song, index, name=None):
        self.song = song
        self.index = index
        self.name = '%s-%i' % (song.name if name is None else name, index)

    def load(self):
        if type(self.song) is Song:
            return self.song.load()
        else:
            return load_song(self.song)
