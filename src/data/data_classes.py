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
    wavelets: bool
    mother_wavelet: str
    wavelet_scale: int


class _Segment(TypedDict):
    track_id: int
    segment_id: int
    iq_matrix: np.ndarray[np.complex]
    doppler_burst: np.ndarray[np.int]
    geolocation_id: int
    geolocation_type: str
    sensor_id: int
    snr_type: str
    date_index: int
    target_type: str
    is_validation: bool
    usable: bool


class Segment(object):
    def __init__(self, config: _Config, **kwargs):
        self.config = config
        self.segment_dict = _Segment(**kwargs)

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

    def iq_to_spectogram(self, axis=0, doppler: bool = False):
        """
        Calculates spectrogram of 'iq_sweep_burst'.

        Arguments:
            iq_burst -- {ndarray} -- 'iq_sweep_burst' array
            axis -- {int} -- axis to perform DFT in (Default = 0)

        Returns:
        Transformed iq_burst array
        """
        iq_burst = self.segment_dict['iq_matrix']
        iq = np.log(np.abs(np.fft.fft(self.hann(iq_burst), axis=axis)))
        iq = np.maximum(np.median(iq) - 1, iq)
        return iq

    def iq_to_scalogram(self, flip: bool = False):
        """
        calculate a scalogram matrix that preforms a continues wavelet transformation on the data.
        return a 3-d array that keeps the different scaled scalograms as different channels

        Arguments:
            iq_matrix (array-like): array of complex signal data, rows represent spatial location, columns time
            flip (bool): optional argument for flipping the row order of the matrix.
            transformation (string): name of wavelet signal to use as mother signal. default to gaussian kernel
        return:
            3-d scalogram: array like transformation that correspond with correlation of different frequency wavelets at different time-points.

        1. select each column of the IQ matrix
        2. apply hann-window smoothing
        3. preform Continues Wavelet Transformation (data, array of psooible scale values, type of transformation)
        """

        scalograms = []
        # analyze each column (time-point) seperatly.
        iq_matrix = self.normalize(self.segment_dict['iq_matrix'])
        # TODO see if this can be vectorized with np.apply_along_axis
        for j in range(iq_matrix.shape[1]):
            # preform hann smoothing on a column - results in a singal j-2 sized column
            # preform py.cwt transformation, returns coefficients and frequencies

            coef, freqs = pywt.cwt(self.hann(iq_matrix[:, j][:, np.newaxis]),
                                   np.arange(1, self.config.get('wavelet_scale', 9)),
                                   self.config.get('mother_wavelet', 'cgau1'))
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

    def flip_scalogram_horizontal(self):
        # TODO function to flip segment-as-scalogram horizontally
        assert False

    def flip_scalogram_vertical(self):
        # TODO function to flip segment-as-scalogram vertically
        assert False


class _Track(TypedDict):
    track_id: int
    iq_sweep_burst: np.ndarray[np.complex]
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


class _SubTrack(TypedDict):
    tracks: List[Union[np.ndarray[np.complex], np.ndarray]]
    bursts: List[np.ndarray[np.int]]
    labels: List[np.ndarray[np.str]]

class _Segments(TypedDict):
    segments: List[Union[np.ndarray[np.complex], np.ndarray]]
    bursts: List[np.ndarray[np.int]]
    labels: List[np.ndarray[np.str]]
    segment_ids: List[np.ndarray[np.int]]

# Make the Track Class based on torch Dataset to support iterable dataset
class TrackDS(Dataset):
    """Dataset for a batch of segments from one track."""

    def __init__(self, track_id: int, track: _Track, config: _Config,
                 sub_tracks: _SubTrack = None, segment_list: _Segments = None):
        """
        Arguments:
            data -- {dict} -- data for
        """
        self.track_id = track_id
        self.config = config
        self.track = track
        self.sub_tracks = sub_tracks
        self.segment_list = segment_list

    def __len__(self):
        return len(self.track)

    def __getitem__(self, idx):
        assert False

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
        bursts = []
        labels = []
        for i, use in enumerate(self.track['usable']):
            if not use:
                if self.track['usable'][previous_i]:
                    if i - previous_i > 0:
                        start = previous_i * 32
                        end = i * 32
                        tracks.append(self.track['iq_sweep_burst'][:, start:end])
                        bursts.append(self.track['doppler_burst'][start:end])
                        labels.append(self.track['target_type'][i])
                previous_i = i + 1
        if not tracks:
            tracks.append(self.track['iq_sweep_burst'])
            bursts.append(self.track['doppler_burst'])
            labels.append(self.track['target_type'])
        self.sub_tracks = {'tracks': tracks, 'bursts': bursts, 'labels': labels}

    @staticmethod
    def split_2d_array(track: np.ndarray, shift_segment: int = 1) -> List[np.ndarray]:
        """
        Splits a 2D track (IQ Matrix or spectogram) into N new segments:
            formula (track.shape[1]) - 32)/shift_segment.

        Arguments:
        track -- {ndarray} -- spectogram/IQ matrix, dimensions (>32, 128)
        shift_segment -- {int} -- Size of step to shift track to generate new segments

        Returns:
        List new segments [segment_array, segment_array]
        """
        indices = range(0, track.shape[1] - 32, shift_segment)
        segments = []
        for i in indices:
            segment = track[:, i: i + 32].copy()
            segments.append(segment)
        return segments

    @staticmethod
    def split_3d_array(self):
        # TODO implement function like split_2d_track for 3d scalogram
        assert False

    @staticmethod
    def split_1d_array(burst: np.ndarray, shift_segment: int = 1) -> List[list]:
        """
        Splits a doppler burst into N new segments
            formula (len(burst) - 32)/shift_segment.

        Arguments:
        burst -- {ndarray} -- array with dimensions (>32,)
        shift_segment -- {int} -- Size of step to shift track to generate new segments

        Returns:
        Dictionary of new segments like {burst_index: new_burst}
        """
        indices = range(0, len(burst) - 32, shift_segment)
        bursts = []
        for i in indices:
            new_burst = burst[i: i + 32].copy()
            bursts.append(new_burst)
        return bursts

    def create_new_segments_from_splits(self):
        """Splits a list of tracks into new segments of size (128, 32) by shifting the existing track in shift_segment increments
            Returns dictionary with list of segments and corresponding bursts and labels
            Arguments:
                data -- {dict} -- contains keys:  {tracks', 'bursts', 'labels'}
                shift_segment -- {int} -- the number of index steps to move for each segment split
        """
        new_segments = []
        new_bursts = []
        new_labels = []
        for i, track in enumerate(self.sub_tracks['tracks']):
            if track.shape[1] > 32:
                new_segments.extend(self.split_2d_array(track=track, shift_segment=self.config['shift_segment']))
                new_bursts.extend(self.split_1d_array(burst=self.sub_tracks['bursts'][i],
                                                      shift_segment=self.config['shift_segment']))
                new_labels.extend([self.sub_tracks['labels'][i]] * len(new_bursts[-1]))
            else:
                new_segments.extend(track)
                new_bursts.extend(self.sub_tracks['bursts'][i])
                new_labels.append(self.sub_tracks['labels'][i])
        self.segment_list = {'segments': new_segments, 'bursts': new_bursts, 'labels': new_labels}

    def split_iq_matrix(self):
        # TODO function to create new segments from IQ Matrix
        assert False

    def split_spectograms(self):
        # TODO function to create new segments from spectograms
        assert False

    def split_scalograms(self):
        # TODO function to create new segments from scalograms
        assert False


class DataDict(object):
    data_df: Union[Union[DataFrame, Series], Any]
    file_path: str
    config: _Config
    train_dict: dict
    train_dict: List[TrackDS]

    def __init__(self, file_path, config: _Config):
        super().__init__()
        self.file_path = file_path
        self.config = config
        self.train_dict = {}
        self.data_df = pd.DataFrame()
        self.track_objects = []

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
                                               iq_sweep_burst=pd.NamedAgg(column="iq_sweep_burst", aggfunc=list),
                                               doppler_burst=pd.NamedAgg(column="doppler_burst", aggfunc=list),
                                               geolocation_type=pd.NamedAgg(column="geolocation_type", aggfunc=list),
                                               geolocation_id=pd.NamedAgg(column="geolocation_id", aggfunc=list),
                                               sensor_id=pd.NamedAgg(column="sensor_id", aggfunc=list),
                                               snr_type=pd.NamedAgg(column="snr_type", aggfunc=list),
                                               date_index=pd.NamedAgg(column="target_type", aggfunc=list),
                                               segment_id=pd.NamedAgg(column="segment_id", aggfunc=list),
                                               )
        df_tracks['iq_sweep_burst'] = df_tracks['iq_sweep_burst'].apply(lambda x: np.concatenate(x, axis=-1))
        df_tracks['doppler_burst'] = df_tracks['doppler_burst'].apply(lambda x: np.concatenate(x, axis=-1))
        df_tracks['target_type'] = df_tracks['target_type'].apply(np.array)
        df_tracks['usable'] = df_tracks['usable'].apply(np.array)
        df_tracks['geolocation_type'] = df_tracks['geolocation_type'].apply(np.array)
        df_tracks['geolocation_id'] = df_tracks['geolocation_id'].apply(np.array)
        df_tracks['sensor_id'] = df_tracks['sensor_id'].apply(np.array)
        df_tracks['snr_type'] = df_tracks['snr_type'].apply(np.array)
        df_tracks['date_index'] = df_tracks['date_index'].apply(np.array)
        df_tracks['segment_id'] = df_tracks['segment_id'].apply(np.array)
        self.track_objects = [TrackDS(track_id=k, track=v) for k, v in df_tracks.to_dict(orient='index').items()]
