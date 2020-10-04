import random
from typing import Dict, TypedDict, List, Any, Union
import numpy as np
import pandas as pd
import pywt
from pandas import Series, DataFrame
from torch.utils.data import Dataset, IterableDataset, DataLoader

from src.data.get_data import load_data, aux_split, append_dict


class _Config(TypedDict):
    """
    num_tracks -- {int} -- # of tracks to take from aux dataset
    valratio -- {int} -- Ratio of train/val split
    get_shifts -- {bool} -- Flag to add shifts
    shift_segment -- {int} -- How much to shift tracks to generate new segments
    get_horizontal_flip -- {bool} -- Flag to add horizontal flips
    get_vertical_flip -- {bool} -- Flag to add vertical flips
    wavelets -- {bool} -- {bool} -- Flag to transform IQ burst into 3-d 7 channels scalograms
    """
    num_tracks: int
    valratio: int
    get_shifts: bool
    shift_segment: int
    get_horizontal_flip: bool
    get_vertical_flip: bool
    output_data_type: str
    wavelets: bool
    mother_wavelet: str
    wavelet_scale: int
    batch_size: int


class _Segment(TypedDict):
    track_id: int
    segment_id: int
    iq_matrix: np.ndarray[np.complex]
    spectrogram: np.ndarray[np.float]
    scalogram: np.ndarray[np.float]
    doppler_burst: np.ndarray[np.int]
    geolocation_id: int
    geolocation_type: str
    sensor_id: int
    snr_type: str
    date_index: int
    target_type: str
    is_validation: bool
    usable: bool


class _Track(TypedDict):
    track_id: int
    output_array: np.ndarray[Union[np.complex, np.float]]
    doppler_burst: np.ndarray[np.int]
    target_type: np.ndarray[np.str]
    usable: np.ndarray[np.bool]
    geolocation_id: np.ndarray[np.int]
    geolocation_type: np.ndarray[np.str]
    sensor_id: np.ndarray[np.int]
    snr_type: np.ndarray[np.str]
    date_index: np.ndarray[np.int]
    is_validation: np.ndarray[np.bool]
    segment_id: np.ndarray[np.int]
    config: _Config


class Segment(object):
    def __init__(self, config: _Config):
        self.config = config

    @staticmethod
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

    @staticmethod
    def normalize(iq):
        """
        Calculates normalized values for iq_sweep_burst matrix:
        (vlaue-mean)/std.
        """
        m = iq.mean()
        s = iq.std()
        return (iq - m) / s

    @staticmethod
    def iq_to_spectogram(iq_burst, axis=0, doppler: bool = False):
        """
        Calculates spectrogram of 'iq_sweep_burst'.

        Arguments:
            iq_burst -- {ndarray} -- 'iq_sweep_burst' array
            axis -- {int} -- axis to perform DFT in (Default = 0)

        Returns:
        Transformed iq_burst array
        """
        iq = np.log(np.abs(np.fft.fft(Segment.hann(iq_burst), axis=axis)))
        iq = np.maximum(np.median(iq) - 1, iq)
        return iq

    @staticmethod
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
        iq_matrix = Segment.normalize(iq_burst)
        # TODO see if this can be vectorized with np.apply_along_axis
        for j in range(iq_matrix.shape[1]):
            # preform hann smoothing on a column - results in a singal j-2 sized column
            # preform py.cwt transformation, returns coefficients and frequencies

            coef, freqs = pywt.cwt(Segment.hann(iq_matrix[:, j][:, np.newaxis]),
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

    @staticmethod
    def flip_iq_horizontal(iq):
        return iq.real * -1 + 1j*iq.imag

    @staticmethod
    def flip_iq_vertical(iq):
        return iq.real + -1j*iq.imag

    @staticmethod
    def flip_spectogram_horizontal(spectogram):
        return np.flip(spectogram, axis=0)

    @staticmethod
    def flip_spectogram_vertical(spectogram):
        return np.flip(spectogram, axis=1)

    @staticmethod
    def flip_scalogram_horizontal(scalogram):
        return np.flip(scalogram, axis=0)

    @staticmethod
    def flip_scalogram_vertical(scalogram):
        return np.flip(scalogram, axis=1)


# Make the Track Class based on torch Dataset to support iterable dataset
class TrackDS(Dataset):
    """Dataset for a batch of segments from one track."""

    def __init__(self, track_id: int, track: _Track, config: _Config):
        """
        Arguments:
            data -- {dict} -- data for
        """
        self.track_id = track_id
        self.config = config
        self.track = track

    def __len__(self):
        return len(self.track)

    def __getitem__(self, idx):
        assert False
        # TODO implement get a track and convert to output type fetched from config

class Track(object):
    """Dataset for a batch of segments from one track."""

    def __init__(self, track_id: int, track: _Track, config: _Config):
        """
        Arguments:
            data -- {dict} -- data for
        """
        self.track_id = track_id
        self.config = config
        self.track = track
        self.sub_tracks = self.validate_subtracks()
        self.segment_list = [_Track()]
        self.create_new_segments_from_splits()
        random.shuffle(self.segment_list)

    def validate_subtracks(self):
        """This algorithm works on the assumption that we have a Boolean Array in data['usable'] indicating whether a
        given segment is to be used for augmentation -- eligibility is set in get_track_level_data.
        It will create a list of the longest adjacent sub-tracks contained in a track and return the broken-up track as a
        list of lists, along with the corresponding doppler_burst and label arrays as lists of lists in a dictionary.

        Arguments:
                data -- {dict} -- data for one track with parameters: {'iq_sweep_burst', 'doppler_burst',
                                                                        'target_type', 'usable'}
        """
        # shift_segment = config.get('shift_segment', 1)
        previous_i = 0
        tracks = []
        for i, use in enumerate(self.track['usable']):
            if not use:
                if self.track['usable'][previous_i]:
                    if i - previous_i > 0:
                        start = previous_i * 32
                        end = i * 32
                        if self.config['output_data_type'] == 'scalogram':
                            track = self.track['output_array'][:, start:end, :]
                        else:
                            track = self.track['output_array'][:, start:end, :]
                        tracks.append(_Track(track_id=f'{self.track_id}_{previous_i}_{i}',
                                             output_array=track,
                                             doppler_burst=self.track['doppler_burst'][start:end],
                                             target_type=self.track['target_type'][i],
                                             config=self.config))
                previous_i = i + 1
        if not tracks:
            tracks.append(_Track(track_id=self.track_id,
                                 output_array=self.track['output_array'],
                                 doppler_burst=self.track['doppler_burst'],
                                 target_type=self.track['target_type']))
        return tracks


    @staticmethod
    def split_Nd_array(array: np.ndarray, shift_segment: int = 1) -> List[np.ndarray]:
        """
        Splits a N-dimensional Array (Doppler Burst, IQ Matrix, spectogram, scalogram) into K new segments:
            formula (track.shape[N]) - 32)/K.

        Arguments:
        track -- {ndarray} -- spectogram/IQ matrix, dimensions (>32, 128)
        shift_segment -- {int} -- Size of step to shift track to generate new segments
        dim -- {int} -- Array dimension along which to split
        Returns:
        List new segments [segment_array, segment_array]
        """

        if array.ndim == 1:
            indices = range(0, len(array) - 32, shift_segment)
            segments = [np.take(array, np.arange(i, i + 32), axis=0).copy() for i in indices]
        else:
            indices = range(0, array.shape[1] - 32, shift_segment)
            segments = [np.take(array, np.arange(i, i + 32), axis=1).copy() for i in indices]
        return segments

    def create_new_segments_from_splits(self):
        """Splits a list of track objects into new segments of size (128, 32) by shifting the existing track in shift_segment increments
            Returns dictionary with list of segments and corresponding bursts and labels
            Arguments:
                data -- {dict} -- contains keys:  {tracks', 'bursts', 'labels'}
                shift_segment -- {int} -- the number of index steps to move for each segment split
        """
        new_segments = []
        new_bursts = []
        new_labels = []
        for i, track in enumerate(self.sub_tracks):
            if track['output_array'].shape[1] > 32:
                new_segments.extend(self.split_Nd_array(array=track['output_array'],
                                                        shift_segment=self.config['shift_segment']))
                new_bursts.extend(self.split_Nd_array(array=track['doppler_burst'],
                                                      shift_segment=self.config['shift_segment']))
                new_labels.extend([track['target_type'][i]] * len(new_bursts[-1]))
            else:
                new_segments.extend(track)
                new_bursts.extend(tracks['bursts'][i])
                new_labels.append(tracks['labels'][i])
        self.segment_list = {'segments': new_segments, 'bursts': new_bursts, 'labels': new_labels}


class DataDict(IterableDataset):
    data_df: Union[Union[DataFrame, Series], Any]
    file_path: str
    config: _Config
    train_dict: dict
    track_objects: List[TrackDS]
    output_data_type: str

    def __init__(self, file_path, config: _Config):
        super().__init__()
        self.file_path = file_path
        self.config = config
        self.train_dict = {}
        self.data_df = pd.DataFrame()
        self.track_objects = []
        self.create_track_objects()
        self.output_type = self.config.get('output_data_type', 'spectrogram')

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
        self.train_dict = append_dict(training_dict, train_aux)

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
        df = pd.DataFrame.from_dict(self.train_dict, orient='index').transpose()
        df = self.split_train_val_as_pd(data=df, ratio=self.config.get('valratio', 6))
        df.sort_values(by=['track_id', 'segment_id'], inplace=True)

        # validating that each track consists of segments with same values in following columns
        columns_to_check = ['geolocation_type', 'geolocation_id', 'sensor_id', 'snr_type', 'date_index', 'target_type']
        conditions = [(df.groupby('track_id')[col].shift(0) == df.groupby('track_id')[col].shift(1).bfill())
                      for col in columns_to_check]
        df['usable'] = np.select(conditions, conditions, default=False)
        df.loc[df['is_validation'] == True, 'usable'] = False
        df.loc[df['is_validation'].shift(1).bfill() == True, 'usable'] = False
        # only keep target_type, usable, iq_sweep_burst and doppler_burst_column
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
            df_tracks['output_array'] = df_tracks['output_array'].apply(Segment.iq_to_scalogram,
                                                                        self.config['mother_wavelet'],
                                                                        self.config['wavelet_scale'])
        else:
            df_tracks['output_array'] = df_tracks['output_array'].apply(Segment.iq_to_spectogram)
        df_tracks['doppler_burst'] = df_tracks['doppler_burst'].apply(lambda x: np.concatenate(x, axis=-1))
        df_tracks['target_type'] = df_tracks['target_type'].apply(np.array)
        df_tracks['usable'] = df_tracks['usable'].apply(np.array)
        df_tracks['geolocation_type'] = df_tracks['geolocation_type'].apply(np.array)
        df_tracks['geolocation_id'] = df_tracks['geolocation_id'].apply(np.array)
        df_tracks['sensor_id'] = df_tracks['sensor_id'].apply(np.array)
        df_tracks['snr_type'] = df_tracks['snr_type'].apply(np.array)
        df_tracks['date_index'] = df_tracks['date_index'].apply(np.array)
        df_tracks['segment_id'] = df_tracks['segment_id'].apply(np.array)
        self.track_objects = df_tracks.to_dict(orient='index')

    def process_tracks(self):
        block_size = self.config.get('batch_size', 50)
        loaders = []
        for i, track in self.track_objects:
            data = TrackDS(track_id=i, track=track, config=self.config)
            loaders.append(DataLoader(data, batch_size=block_size, shuffle=True))
        return loaders
