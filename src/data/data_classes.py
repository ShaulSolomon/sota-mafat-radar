from typing import Dict, TypedDict, List, Any, Union
import numpy as np
import pandas as pd
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
    def __init__(self, **kwargs):
        self.segment_dict = _Segment(**kwargs)

    def iq_to_spectogram(self, doppler: bool = False):
        # TODO function for turning IQ matrix into spectogram, with option of adding doppler burst
        assert False

    def iq_to_scalogram(self):
        # TODO function for turning IQ matrix into scalogram
        assert False

    def flip_iq_horizontal(self):
        # TODO function to flip segment-as-IQ matrix horizontally
        assert False

    def flip_iq_vertical(self):
        # TODO function to flip segment-as-IQ matrix vertically
        assert False

    def flip_spectogram_horizontal(self):
        # TODO function to flip segment-as-spectogram horizontally
        assert False

    def flip_spectogram_vertical(self):
        # TODO function to flip segment-as-spectogram vertically
        assert False

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


# Make the Track Class based on torch Dataset to support iterable dataset
class TrackDS(Dataset):
    """Dataset for a batch of segments from one track."""

    def __init__(self, track_id: int, track: _Track):
        """
        Arguments:
            data -- {dict} -- data for
        """
        self.track_id = track_id
        self.track = track

    def __len__(self):
        return len(self.track)

    def __getitem__(self, idx):
        return self.track[idx]

    def validate_subtracks(self):
        # TODO function to split track into sub-tracks according to the is_validation attribute of sequential segments.
        assert False

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
