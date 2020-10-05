from itertools import chain, cycle
from typing import TypedDict, List, Union, Dict

import pandas as pd
import numpy as np
import random

import pywt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, IterableDataset
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd()))))
from src.data.get_data import append_dict, load_data, aux_split

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


def iq_to_spectogram(iq_burst, axis=0, doppler: bool = False):
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


def iq_to_scalogram(iq_burst, flip: bool = False, transformation: str = 'cgau1', scale: int = 9):
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

    scalograms = []
    # analyze each column (time-point) seperatly.
    iq_matrix = normalize(iq_burst)
    # TODO see if this can be vectorized with np.apply_along_axis
    for j in range(iq_matrix.shape[1]):
        # preform hann smoothing on a column - results in a singal j-2 sized column
        # preform py.cwt transformation, returns coefficients and frequencies

        coef, freqs = pywt.cwt(hann(iq_matrix[:, j][:, np.newaxis]),
                               np.arange(1, scale), transformation)
        # coefficient matrix returns as a (num_scalers-1, j-2 , 1) array, transform it into a 2-d array

        if flip:
            coef = np.flip(coef, axis=0)
        # log normalization of the data
        coef = np.log(np.abs(coef))
        # first column correspond to the scales, rest is the coefficients
        coef = coef[:, :, 0]

        scalograms.append(coef)

    stacked_scalogram = np.stack(scalograms)
    stacked_scalogram = np.maximum(np.median(stacked_scalogram) - 1., stacked_scalogram)
    stacked_scalogram = np.transpose(stacked_scalogram, (2, 0, 1))
    return stacked_scalogram


def split_Nd_array(array: np.ndarray) -> List[np.ndarray]:
    """
    Splits a N-dimensional Array (Doppler Burst, IQ Matrix, spectogram, scalogram) into K new segments:
        formula (track.shape[N]) - 32)/K.

    Arguments:
    track -- {ndarray} -- spectogram/IQ matrix, dimensions (>32, 128)
    # shift_segment -- {int} -- Size of step to shift track to generate new segments
    dim -- {int} -- Array dimension along which to split
    Returns:
    List new segments [segment_array, segment_array]
    """

    if array.ndim == 1:
        indices = range(0, len(array) - 32)
        segments = [np.take(array, np.arange(i, i + 32), axis=0).copy() for i in indices]
    else:
        indices = range(0, array.shape[1] - 32)
        segments = [np.take(array, np.arange(i, i + 32), axis=1).copy() for i in indices]
    return segments


class _Segment(TypedDict):
    segment_id: Union[int, str]
    output_array: np.ndarray
    doppler_burst: np.ndarray
    target_type: str


class TrackDS(Dataset):
    """Dataset for a batch of segments from one track."""
    segment_list: List[_Segment]

    def __init__(self, segment_list):
        """
        Arguments:
            data -- {dict} -- data for
        """
        self.segment_list = segment_list

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, idx):
        return self.segment_list[idx]


class Config(TypedDict):
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
    mother_wavelet: str
    wavelet_scale: int
    batch_size: int


def create_new_segments_from_splits(segment_list: List[_Segment]) -> List[_Segment]:
    """Splits a list of tracks into new segments of size (128, 32) by shifting the existing track increments of 1 along slow-time axis
        Returns dictionary with list of segments and corresponding bursts and labels
        Arguments:
            data -- {dict} -- contains keys:  {tracks', 'bursts', 'labels'}
    """
    new_segments = []
    for i, segment in enumerate(segment_list):
        if segment['output_array'].shape[1] > 32:
            output_array = split_Nd_array(array=segment['output_array'])
            bursts = split_Nd_array(array=segment['doppler_burst'])
            labels = [segment['target_type']] * len(bursts)
            new_segments.extend([_Segment(segment_id=f'{i}_{j}',
                                          output_array=array,
                                          doppler_burst=bursts[j],
                                          target_type=labels[j]) for j, array in enumerate(output_array)])

        else:
            new_segments.append(segment)
    return new_segments


def create_flipped_segments(segment_list: List[_Segment], flip_type: str = 'vertical') -> List[_Segment]:
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
    print(f'Flipping segments {flip_type}')
    for i, segment in enumerate(tqdm(segment_list)):
        flipped_segments.append(_Segment(segment_id=f'{segment["segment_id"]}_{i}',
                                         output_array=flip(segment["output_array"]),
                                         doppler_burst=burst_flip(segment['doppler_burst']),
                                         target_type=segment['target_type']))
    return flipped_segments


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

        # validating that each track consists of segments with same values in following columns
        columns_to_check = ['geolocation_type', 'geolocation_id', 'sensor_id', 'snr_type', 'date_index', 'target_type']
        conditions = [(df.groupby('track_id')[col].shift(0) == df.groupby('track_id')[col].shift(1).bfill())
                      for col in columns_to_check]
        df['usable'] = np.select(conditions, conditions, default=False)
        df.loc[df['is_validation'], 'usable'] = False
        df.loc[df['is_validation'].shift(1).bfill(), 'usable'] = False
        # df['subtrack_id'] = df.groupby('track_id')['segment_id'].transform('rank')
        # df.groupby((df['usable'].shift() != df['usable']).cumsum())

        # save validation segments to object and drop from current DF
        val_df = df.loc[df.is_validation].copy().set_index('segment_id')
        df = df.loc[~df.is_validation].copy()

        df_tracks = df.groupby('track_id').agg(target_type=pd.NamedAgg(column="target_type", aggfunc=list),
                                               usable=pd.NamedAgg(column="usable", aggfunc=list),
                                               output_array=pd.NamedAgg(column="iq_sweep_burst", aggfunc=list),
                                               doppler_burst=pd.NamedAgg(column="doppler_burst", aggfunc=list),
                                               geolocation_type=pd.NamedAgg(column="geolocation_type", aggfunc=list),
                                               geolocation_id=pd.NamedAgg(column="geolocation_id", aggfunc=list),
                                               sensor_id=pd.NamedAgg(column="sensor_id", aggfunc=list),
                                               snr_type=pd.NamedAgg(column="snr_type", aggfunc=list),
                                               date_index=pd.NamedAgg(column="target_type", aggfunc=list),
                                               segment_id=pd.NamedAgg(column="segment_id", aggfunc=list),
                                               )
        df_tracks['output_array'] = df_tracks['output_array'].apply(lambda x: np.concatenate(x, axis=-1))

        if self.output_data_type == 'scalogram':
            print('Converting IQ matricies to Scalogram')
            df_tracks['output_array'] = df_tracks['output_array'].progress_apply(iq_to_scalogram,
                                                                                 self.config['mother_wavelet'],
                                                                                 self.config['wavelet_scale'])
            val_df['output_array'] = val_df['iq_sweep_burst'].progress_apply(iq_to_scalogram,
                                                                             self.config['mother_wavelet'],
                                                                             self.config['wavelet_scale'])
        else:
            print('Converting IQ matricies to Spectrogram')
            df_tracks['output_array'] = df_tracks['output_array'].progress_apply(iq_to_spectogram)
            val_df['output_array'] = val_df['iq_sweep_burst'].progress_apply(iq_to_spectogram)

        df_tracks['doppler_burst'] = df_tracks['doppler_burst'].apply(lambda x: np.concatenate(x, axis=-1))
        df_tracks['target_type'] = df_tracks['target_type'].apply(np.array)
        df_tracks['usable'] = df_tracks['usable'].apply(np.array)
        df_tracks['geolocation_type'] = df_tracks['geolocation_type'].apply(np.array)
        df_tracks['geolocation_id'] = df_tracks['geolocation_id'].apply(np.array)
        df_tracks['sensor_id'] = df_tracks['sensor_id'].apply(np.array)
        df_tracks['snr_type'] = df_tracks['snr_type'].apply(np.array)
        df_tracks['date_index'] = df_tracks['date_index'].apply(np.array)
        df_tracks['segment_id'] = df_tracks['segment_id'].apply(np.array)
        return df_tracks.to_dict(orient='index'), val_df.to_dict(orient='index')


def filter_usable_segments(data, output_data_type) -> List[_Segment]:
    """This algorithm works on the assumption that we have a Boolean Array in data['usable'] indicating whether a
    given segment is to be used for augmentation -- eligibility is set in get_track_level_data.
    It will create a list of the longest adjacent sub-tracks contained in a track and return the broken-up track as a
    list of lists, along with the corresponding doppler_burst and label arrays as lists of lists in a dictionary.

    Arguments:
            data -- {dict} -- data for one track with parameters: {'iq_sweep_burst', 'doppler_burst',
                                                                    'target_type', 'usable'}
    """
    track = data['iq_sweep_burst']
    burst = data['doppler_burst']
    previous_i = 0
    tracks = []
    for i, use in enumerate(data['usable']):
        if not use:
            if data['usable'][previous_i]:
                if i - previous_i > 0:
                    start = previous_i * 32
                    end = i * 32
                    if output_data_type == 'scalogram':
                        output_array = track[:, start:end, :]
                    else:
                        output_array = track[:, start:end]

                    tracks.append(_Segment(segment_id=i,
                                           output_array=output_array,
                                           doppler_burst=burst[start:end],
                                           target_type=data['target_type'][i]))
            previous_i = i + 1
    if not tracks:
        if track.shape[1] == 32:
            tracks.append(_Segment(segment_id=data['segment_id'],
                                   output_array=track,
                                   doppler_burst=burst,
                                   target_type=data['target_type']))
    return tracks


class StreamingDataset(IterableDataset):
    data: Dict[int, dict]
    config: Config
    random_index: List[int]
    is_val: bool

    def __init__(self, dataset, config, is_val=False):
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
        self.random_index = list(self.data.keys())
        self.is_val = is_val
        random.shuffle(self.random_index)

    def segments_generator(self, data_dict) -> DataLoader:
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
        filtered_data = filter_usable_segments(data=data_dict, output_data_type=self.config['output_data_type'])

        if self.config.get('get_shifts'):
            resplit_data = create_new_segments_from_splits(filtered_data)
        else:
            resplit_data = filtered_data

        if self.config.get('get_vertical_flip'):
            flips = create_flipped_segments(resplit_data, flip_type='vertical')
            resplit_data = resplit_data + flips
        if self.config.get('get_horizontal_flip'):
            flips = create_flipped_segments(resplit_data, flip_type='horizontal')
            resplit_data = resplit_data + flips

        return DataLoader(TrackDS(resplit_data), batch_size=1, shuffle=True)

    def get_segment_stream(self):
        return chain.from_iterable(map(self.segments_generator, self.data))

    def get_segment_streams(self):
        return zip(*[self.get_segment_stream() for _ in range(self.config['batch_size'])])

    @classmethod
    def split_track_dataset(cls, data_list, batch_size, max_workers):

        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        split_size = batch_size // num_workers

        return [cls(dataset=data_list, config=cls.config) for _ in range(num_workers)]

    def __iter__(self):
        return self.get_segment_streams()


class MultiStreamDataLoader:
    def __init__(self, datasets):
        self.datasets = datasets

    def get_stream_loaders(self):
        return zip(*[DataLoader(dataset=dataset, num_workers=1, batch_size=None) for dataset in self.datasets])

    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            yield list(chain(*batch_parts))
