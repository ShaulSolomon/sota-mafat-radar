from abc import ABC
from collections import defaultdict
from itertools import chain, cycle
from typing import List, Union, Dict, Generator, Any

import pandas as pd
import numpy as np
import random

import pywt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, IterableDataset
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd()))))
from src.data.get_data import append_dict, load_data, aux_split
from src.features.specto_feat import max_value_on_doppler

tqdm.pandas()


def flip_horizontal(array):
    return np.flip(array, axis=0)


def flip_vertical(array):
    return np.flip(array, axis=1)


def return_unchanged(f):
    return f


def burst_vertical_flip(burst):
    return np.abs(128 - burst)


def hann(iq, window=None):
    """
    Hann smoothing of 'iq_sweep_burst'.

    Arguments:
      iq -- {ndarray} -- 'iq_sweep_burst' array
      window -- Range of Hann window indices (Default=None)
        If None the whole column is taken
      Returns:
        Regularized iq shaped as (window[1] - window[0] - 2, iq.shape[1])
      """
    if window is None:
        window = [0, len(iq)]

    N = window[1] - window[0] - 1
    n = np.arange(window[0], window[1])
    n = n.reshape(len(n), 1)
    hannCol = 0.5 * (1 - np.cos(2 * np.pi * (n / N)))
    return (hannCol * iq[window[0]:window[1]])[1:-1]


def normalize(iq):
    """
    Calculates normalized values for iq_sweep_burst matrix:
    (vlaue-mean)/std.
    """
    m = iq.mean()
    s = iq.std()
    return (iq - m) / s


def iq_to_spectogram(iq_burst, axis=0):
    """
    Calculates spectrogram of 'iq_sweep_burst'.

    Arguments:
        iq_burst -- {ndarray} -- 'iq_sweep_burst' array
        axis -- {int} -- axis to perform DFT in (Default = 0)

    Returns:
    Transformed iq_burst array
    """
    iq = np.log(np.abs(np.fft.fft(hann(iq_burst), axis=axis)))
    iq = np.maximum(np.median(iq) - 1, iq)
    return iq


def my_cwt(iq_matrix_col, transformation: str = 'cgau1', scale: int = 9):
    coef, _ = pywt.cwt(hann(iq_matrix_col[:, np.newaxis]),
                       np.arange(1, scale), transformation)
    coef = np.log(np.abs(coef))
    coef = coef[:, :, 0]
    return coef


def iq_to_scalogram(iq_burst, transformation: str = 'cgau1', scale: int = 9):
    """
    calculate a scalogram matrix that preforms a continues wavelet transformation on the data.
    return a 3-d array that keeps the different scaled scalograms as different channels

    Arguments:
        iq_matrix (array-like): array of complex signal data, rows represent spatial location, columns time
        flip (bool): optional argument for flipping the row order of the matrix.
        transformation (string): name of wavelet signal to use as mother signal. default to gaussian kernel
        scale (int): number of scales to apply wavelet over
    return:
        3-d scalogram: array like transformation that correspond with correlation of different frequency wavelets at different time-points.

    1. select each column of the IQ matrix
    2. apply hann-window smoothing
    3. preform Continues Wavelet Transformation (data, array of psooible scale values, type of transformation)
    :param mother_wavelet:
    """

    iq_matrix = normalize(iq_burst)
    scalograms = np.apply_along_axis(my_cwt, 0, iq_matrix, transformation=transformation, scale=scale)

    stacked_scalogram = np.stack(scalograms)
    stacked_scalogram = np.maximum(np.median(stacked_scalogram) - 1., stacked_scalogram)
    stacked_scalogram = np.transpose(stacked_scalogram, (1, 2, 0))
    return stacked_scalogram


def split_Nd_array(array: np.ndarray, nsplits: int) -> List[np.ndarray]:
    """
    Splits a N-dimensional Array (Doppler Burst, IQ Matrix, spectogram, scalogram) into new segments:
        new segment count = (track.shape[N] - (32/K))/K

    Arguments:
    track -- {ndarray} -- spectogram/IQ matrix, dimensions (>32, 128)
    k -- {int} -- Size of step to shift track to generate new segments
    dim -- {int} -- Array dimension along which to split
    Returns:
    List new segments [segment_array, segment_array]
    """

    if array.ndim == 1:
        indices = range(0, len(array) - 31, nsplits)
        segments = [np.take(array, np.arange(i, i + 32), axis=0).copy() for i in indices]
    else:
        indices = range(0, array.shape[1] - 31, nsplits)
        segments = [np.take(array, np.arange(i, i + 32), axis=1).copy() for i in indices]
    return segments


class _Segment(Dict, ABC):
    segment_id: Union[int, str]
    output_array: np.ndarray
    doppler_burst: np.ndarray
    target_type: np.ndarray
    segment_count: int

    def assert_valid_spectrogram(self):
        assert isinstance(self['segment_id'], (
        int, str)), f'Segment id must be int or string: is {type(self["output_array"])} with {self["segment_id"]}'
        assert isinstance(self['output_array'],
                          np.ndarray), f'Output array must be nd.array: is {type(self["output_array"])} with segment id {self["segment_id"]}'
        assert self['output_array'].shape == (126,
                                              32), f'Output array must have shape (126, 32): is {self["output_array"].shape} with segment id {self["segment_id"]}'
        assert isinstance(self['doppler_burst'],
                          np.ndarray), f'Doppler burst must be nd.array: is {type(self["doppler_burst"])} with segment id {self["segment_id"]}'
        assert self['doppler_burst'].shape == (
        32,), f'Doppler burst must have shape (32,): is {self["doppler_burst"].shape} with segment id {self["segment_id"]}'
        assert isinstance(self['target_type'],
                          int), f'Target type must be np.bool_: is {type(self["target_type"])} with segment id {self["segment_id"]}'

    def assert_valid_scalogram(self):
        assert isinstance(self['segment_id'], (
        int, str)), f'Segment id must be int or string: is {type(self["output_array"])} with {self["segment_id"]}'
        assert isinstance(self['output_array'],
                          np.ndarray), f'Output array must be nd.array: is {type(self["output_array"])} with segment id {self["segment_id"]}'
        assert self[
                   'output_array'].ndim == 3, f'Output array must b 3D: is {self["output_array"].ndim} with segment id {self["segment_id"]}'
        assert isinstance(self['doppler_burst'],
                          np.ndarray), f'Doppler burst must be nd.array: is {type(self["doppler_burst"])} with segment id {self["segment_id"]}'
        assert self['doppler_burst'].shape == (
        32,), f'Doppler burst must have shape (32,): is {self["doppler_burst"].shape} with segment id {self["segment_id"]}'
        assert isinstance(self['target_type'],
                          int), f'Target type must be nd.array: is {type(self["target_type"])} with segment id {self["segment_id"]}'


class Config(Dict, ABC):
    """
    num_tracks -- {int} -- # of tracks to take from aux dataset
    valratio -- {int} -- Ratio of train/val split
    get_shifts -- {bool} -- Flag to add shifts
    shift_segment -- {int} -- How much to shift tracks to generate new segments
    get_horizontal_flip -- {bool} -- Flag to add horizontal flips
    get_vertical_flip -- {bool} -- Flag to add vertical flips
    wavelets -- {bool} -- {bool} -- Flag to transform IQ burst into 3-d 7 channels scalograms
    """
    file_path: str
    num_tracks: int
    valratio: int
    get_shifts: bool
    get_horizontal_flip: bool
    get_vertical_flip: bool
    output_data_type: str
    include_doppler: bool
    mother_wavelet: str
    wavelet_scale: int
    batch_size: int


class DataDict(object):
    config: Config
    file_path: str
    output_data_type: str
    data_df: pd.DataFrame
    train_data: Dict[int, dict]
    train_size: int
    val_data: Dict[int, dict]
    val_size: int

    def __init__(self, config):
        self.config = config
        self.file_path = self.config.get('file_path')
        self.output_data_type = self.config.get('output_data_type', 'spectrogram')
        self.data_df = self.load_all_datasets()
        self.train_data, self.val_data = self.create_track_objects()

    def load_all_datasets(self):
        """Load all datasets into one dictionary

            Arguments:
                self.file_path -- {str}: path to the data directory, comes from credentials.ini file
                self.config -- {dict}:
                    num_tracks -- {int} -- # of tracks to take from aux dataset

            Returns:
            None
        """

        ##############################
        #####  LOAD RAW DATA    ######
        ##############################

        experiment_auxiliary = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
        experiment_auxiliary_df = load_data(experiment_auxiliary, self.file_path)

        train_aux = aux_split(experiment_auxiliary_df, numtracks=self.config.get('num_tracks', 3))

        train_path = 'MAFAT RADAR Challenge - Training Set V1'
        training_dict = load_data(train_path, self.file_path)

        # Adding segments from the experiment auxiliary set to the training set
        return pd.DataFrame.from_dict(append_dict(training_dict, train_aux), orient='index').transpose()

    @staticmethod
    def split_train_val_as_pd(data, ratio=6):
        """
        Split the data to train and validation set.
        The validation set is built from training set segments of
        geolocation_id 1 and 4.
        Use the function only after the training set is complete and preprocessed.

        Arguments:
          data -- {pandas} -- the data set to split as pandas dataframe
          ratio -- {int} -- ratio to make the split by

        Returns:
          same dataset as input, with adding is_validation column
        """

        data['is_validation'] = ((data['geolocation_id'] == 4) | (data['geolocation_id'] == 1)) & \
                                (data['segment_id'] % ratio == 0)

        return data

    def create_track_objects(self):
        """Transform a dictionary of segment-level datasets into a track-level dictionary

                Arguments:
                    data_dict -- {dict}: contains the data for all segments
                    config -- {dict}:
                        num_tracks -- {int} -- # of tracks to take from aux dataset

                Returns:
                self.data_dict with concatenated I/Q matrix, doppler burst vector, usability index and segment index vector
                """
        # Creating a dataframe to make it easier to validate sequential segments
        df = self.data_df
        df = self.split_train_val_as_pd(data=df, ratio=self.config.get('valratio', 6))
        df.sort_values(by=['track_id', 'segment_id'], inplace=True)
        df.replace({'animal': 0, 'human': 1}, inplace=True)
        df['target_type'] = df['target_type'].astype(int)
        # validating that each track consists of segments with same values in following columns
        columns_to_check = ['geolocation_type', 'geolocation_id', 'sensor_id', 'snr_type', 'date_index', 'target_type']
        conditions = [(df.groupby('track_id')[col].shift(0) == df.groupby('track_id')[col].shift(1).bfill())
                      for col in columns_to_check]
        df['usable'] = np.select(conditions, conditions, default=False)
        df.loc[df['is_validation'], 'usable'] = False
        df.loc[df['is_validation'].shift(1).fillna(False), 'usable'] = False
        df['usable'] = ~df['usable']
        # save validation segments to object and drop from current DF
        val_df = df.loc[df.is_validation].copy().set_index(['track_id', 'segment_id'])
        df = df.loc[~df.is_validation].copy()
        # Creating a subtrack id for immediate grouping into contiguous segments
        df['subtrack_id'] = df.groupby('track_id').usable.cumsum()
        df_tracks = df.groupby(['track_id', 'subtrack_id']).agg(
            target_type=pd.NamedAgg(column="target_type", aggfunc='unique'),
            output_array=pd.NamedAgg(column="iq_sweep_burst", aggfunc=list),
            doppler_burst=pd.NamedAgg(column="doppler_burst", aggfunc=list),
            segment_count=pd.NamedAgg(column="segment_id", aggfunc='count'),
            )
        val_tracks = val_df.groupby(['track_id', 'segment_id']).agg(
            target_type=pd.NamedAgg(column="target_type", aggfunc='unique'),
            output_array=pd.NamedAgg(column="iq_sweep_burst", aggfunc=list),
            doppler_burst=pd.NamedAgg(column="doppler_burst", aggfunc=list),
            segment_count=pd.NamedAgg(column="target_type", aggfunc='count'),
            )
        df_tracks['target_type'] = df_tracks['target_type'].apply(lambda x: x[0])
        val_tracks['target_type'] = val_tracks['target_type'].apply(lambda x: x[0])
        df_tracks['doppler_burst'] = df_tracks['doppler_burst'].apply(lambda x: np.concatenate(x, axis=-1))
        val_tracks['doppler_burst'] = val_tracks['doppler_burst'].apply(lambda x: np.concatenate(x, axis=-1))
        df_tracks['output_array'] = df_tracks['output_array'].apply(lambda x: np.concatenate(x, axis=1))
        val_tracks['output_array'] = val_tracks['output_array'].apply(lambda x: np.concatenate(x, axis=1))
        if self.output_data_type == 'scalogram':
            print('Converting IQ matricies to Scalogram')
            df_tracks['output_array'] = df_tracks['output_array'].progress_apply(iq_to_scalogram,
                                                                                 transformation=self.config[
                                                                                     'mother_wavelet'],
                                                                                 scale=self.config['scale'])
            val_tracks['output_array'] = val_tracks['output_array'].progress_apply(iq_to_scalogram,
                                                                                   transformation=self.config[
                                                                                       'mother_wavelet'],
                                                                                   scale=self.config['scale'])
        else:
            print('Converting IQ matricies to Spectrogram')
            df_tracks['output_array'] = df_tracks['output_array'].progress_apply(iq_to_spectogram)
            val_tracks['output_array'] = val_tracks['output_array'].progress_apply(iq_to_spectogram)

            if self.config.get('include_doppler'):
                df_tracks['output_array'] = df_tracks.progress_apply(lambda row: max_value_on_doppler(row['output_array'], row['doppler_burst']), axis=1)
                val_tracks['output_array'] = val_tracks.progress_apply(lambda row: max_value_on_doppler(row['output_array'], row['doppler_burst']), axis=1)
            df_tracks['output_array'] = df_tracks['output_array'].progress_apply(normalize)
            val_tracks['output_array'] = val_tracks['output_array'].progress_apply(normalize)

        train_segments = [_Segment(segment_id=f'{k[0]}_{k[1]}', **v) for k, v in
                          df_tracks.to_dict(orient='index').items()]
        val_segments = [_Segment(segment_id=f'{k[0]}_{k[1]}', **v) for k, v in
                        val_tracks.to_dict(orient='index').items()]
        return train_segments, val_segments


def create_new_segments_from_splits(segment: _Segment, nsplits: int) -> List[_Segment]:
    """Splits a list of tracks into new segments of size (128, 32) by shifting the existing track increments of 1 along slow-time axis
        Returns dictionary with list of segments and corresponding bursts and labels
        Arguments:
            data -- {dict} -- contains keys:  {tracks', 'bursts', 'labels'}
    """
    new_segments = []
    if segment['output_array'].shape[1] > 32:
        output_array = split_Nd_array(array=segment['output_array'], nsplits=nsplits)
        bursts = split_Nd_array(array=segment['doppler_burst'], nsplits=nsplits)
        new_segments.extend([_Segment(segment_id=f'{segment["segment_id"]}_{j}',
                                      output_array=array,
                                      doppler_burst=bursts[j],
                                      target_type=segment['target_type'],
                                      segment_count=1)
                             for j, array in enumerate(output_array)])

    else:
        new_segments.append(segment)
    return new_segments


def create_flipped_segments(segment_list: Union[List[_Segment], _Segment], flip_type: str = 'vertical') -> List[
    _Segment]:
    """Returns a list of vertically or horizontally flipped segments
        Returns dictionary with list of flipped segments and correspondingly flipped bursts and labels
        Arguments:
            data -- {dict} -- contains keys:  {tracks', 'bursts', 'labels'}
            flip_type {str} -- indicate whether to perform horizontal or vertical flips
    """
    flip = return_unchanged
    burst_flip = return_unchanged
    if flip_type == 'vertical':
        flip = flip_vertical
        burst_flip = burst_vertical_flip
    if flip_type == 'horizontal':
        flip = flip_horizontal
        burst_flip = np.flip
    flipped_segments = []
    if isinstance(segment_list, list):
        for i, segment in enumerate(segment_list):
            flipped_segments.append(_Segment(segment_id=f'{segment["segment_id"]}_{i}',
                                             output_array=flip(segment["output_array"]).copy(),
                                             doppler_burst=burst_flip(segment['doppler_burst']).copy(),
                                             target_type=segment['target_type'],
                                             segment_count=1))
    elif isinstance(segment_list, dict):
        flipped_segments.append(_Segment(segment_id=f'{segment_list["segment_id"]}_{0}',
                                         output_array=flip(segment_list["output_array"]).copy(),
                                         doppler_burst=burst_flip(segment_list['doppler_burst']).copy(),
                                         target_type=segment_list['target_type'],
                                         segment_count=1))
    return flipped_segments


def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


class TrackDS(Dataset):
    """Dataset for a batch of segments from one track."""

    def __init__(self, segment_list: List[_Segment]):
        self.segment_list = segment_list
        for segment in self.segment_list:
            segment['segment_id'] = int(segment['segment_id'])

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, idx):
        return self.segment_list[idx]


class StreamingDataset(IterableDataset):
    data: List[_Segment]
    config: Config
    random_index: List[int]
    is_val: bool

    def __init__(self, dataset, config, is_val=False, shuffle=True):
        """
             arguments:
             ...
             data_records -- {dict}: dictionary containing records of all concatenated tracks indexed by track_id
             config -- {dict}:
                 num_tracks -- {int} -- # of tracks to take from aux dataset
                 valratio -- {int} -- Ratio of train/val split
                 get_shifts -- {bool} -- Flag to add shifts
                 shift_segment -- {int} -- How much to shift tracks to generate new segments
                 get_horizontal_flip -- {bool} -- Flag to add horizontal flips
                 get_vertical_flip -- {bool} -- Flag to add vertical flips
                 block_size -- {int} -- Max number of samples allowed to be held in a memory
        """
        super().__init__()
        self.data = dataset
        self.config = config
        self.is_val = is_val
        self.segment_blocks = []
        self.track_count = 0
        self.total_tracks = len(self.data) - 1
        segment_count = int(sum([((v['segment_count']-1)*32/self.config.get('shift_segment', 32))+1
                                 for v in self.data]))
        if config.get('get_horizontal_flip'): segment_count *= 2
        if config.get('get_vertical_flip'): segment_count *= 2
        self.segment_count = segment_count
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return self.segment_count

    def segments_generator(self, segment_list: _Segment) -> Union[List[_Segment], None]:
        """
        Generates new and/or augmented segments according to configuration parameters.
        Returns a dictionary containing the merged set of segments indexed by
            Arguments:
                data -- {dict} -- data for one track with parameters: ['segment_id', 'geolocation_type', 'geolocation_id',
                                                                       'sensor_id', 'snr_type',
                                                                       'date_index', 'iq_sweep_burst', 'doppler_burst',
                                                                       'target_type', 'usable']
                config -- {dict}:
                     get_shifts -- {bool} -- Flag to add shifts
                     shift_segment -- {int} -- How much to shift tracks to generate new segments
                     get_horizontal_flip -- {bool} -- Flag to add horizontal flips
                     get_vertical_flip -- {bool} -- Flag to add vertical flips
                     block_size -- {int} -- Max number of samples allowed to be held in a memory
        """
        if self.config.get('get_shifts'):
            segment_list = create_new_segments_from_splits(segment_list, nsplits=self.config['shift_segment'])
        else:
            segment_list = create_new_segments_from_splits(segment_list, nsplits=32)

        if self.config.get('get_vertical_flip'):
            flips = create_flipped_segments(segment_list, flip_type='vertical')
            segment_list.extend(flips)
        if self.config.get('get_horizontal_flip'):
            flips = create_flipped_segments(segment_list, flip_type='horizontal')
            segment_list.extend(flips)

        for segment in segment_list:
            if self.config['output_data_type'] == 'scalogram':
                segment.assert_valid_scalogram()
            else:
                segment.assert_valid_spectrogram()
        # if self.config.get('shuffle_stream'):
        self.segment_blocks.extend(segment_list)
        random.shuffle(self.segment_blocks)
        # else:
        #     return segment_list

    def process_tracks_shuffle(self):
        for i, track in enumerate(self.data):
            self.segments_generator(track)
            if i % self.config.get('tracks_in_memory', 100) == self.config.get('tracks_in_memory', 100):
                yield self.segment_blocks
        yield self.segment_blocks

    def shuffle_stream(self):
        return chain(self.process_tracks_shuffle())

    def linear_stream(self):
        return chain(self.segments_generator(track) for track in self.data)

    def __iter__(self):
        # if self.config.get('shuffle_stream'):
        for segments in chain(self.shuffle_stream()):
            yield from segments
        # else:
        #     for segments in chain(self.linear_stream()):
        #         yield from segments


class MultiStreamDataLoader:
    def __init__(self, datasets, config):
        self.datasets = datasets
        self.config = config

    def get_stream_loaders(self):
        return zip(*[DataLoader(StreamingDataset(dataset=dataset, config=self.config), num_workers=1, batch_size=1)
                     for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield list(chain(*batch_parts))
