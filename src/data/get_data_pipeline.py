# import sys
# sys.path.append('/home/shaul/workspace/GitHub/sota-mafat-radar')
import random
import numpy as np
import os
import pandas as pd
from termcolor import colored
import configparser
import pickle
from src.data import get_data
from src.features import augmentations
from src.features import specto_feat
from src.features import add_data
import psutil

import logging

logger = logging.getLogger()


def load_all_datasets(PATH_DATA: str, config: dict = None) -> dict:
    """Load all datasets into one dictionary file

        Arguments:
            PATH_DATA -- {str}: path to the data directory, comes from credentials.ini file
            config -- {dict}:
                num_tracks -- {int} -- # of tracks to take from aux dataset

        Returns:
        Concatenated I/Q matrix and concatenated doppler burst vector
    """

    ### Default parameter
    num_tracks = config.get('num_tracks', 3)

    ##############################
    #####  LOAD RAW DATA    ######
    ##############################

    experiment_auxiliary = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
    experiment_auxiliary_df = get_data.load_data(experiment_auxiliary, PATH_DATA)
    logger.info(f"experiment_auxiliary:{experiment_auxiliary_df['date_index'].shape}")

    train_aux = get_data.aux_split(experiment_auxiliary_df, numtracks=num_tracks)
    logger.info(f"train_aux:{train_aux['date_index'].shape}")

    train_path = 'MAFAT RADAR Challenge - Training Set V1'
    training_dict = get_data.load_data(train_path, PATH_DATA)

    # Adding segments from the experiment auxiliary set to the training set
    train_dict = get_data.append_dict(training_dict, train_aux)

    logger.info(
        f"training_dict({training_dict['date_index'].shape}) + aux dataset({train_aux['date_index'].shape}) = full train({train_dict['date_index'].shape})")
    return train_dict


def get_track_level_data(data_dict: dict) -> dict:
    """Transform a dictionary of segment-level datasets into a track-level dictionary

            Arguments:
                data_dict -- {dict}: contains the data for all segments
                config -- {dict}:
                    num_tracks -- {int} -- # of tracks to take from aux dataset

            Returns:
            Dictionary with concatenated I/Q matrix, doppler burst vector, usability index and segment index vector
            """
    columns = ['geolocation_type', 'geolocation_id', 'sensor_id', 'snr_type', 'date_index', 'target_type']
    # Creating a dataframe to make it easier to validate sequential segments
    df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
    df.sort_values(by=['track_id', 'segment_id'], inplace=True)
    conditions = [(df.groupby('track_id')[col].shift(0) == df.groupby('track_id')[col].shift(1).bfill())
                  for col in columns]
    df['usable'] = np.select(conditions, conditions, default=False)
    df.loc[df['is_validation'] == True, 'usable'] = False
    df.loc[df['is_validation'].shift(1).bfill() == True, 'usable'] = False
    df_tracks = df.groupby('track_id').agg(target_type=pd.NamedAgg(column="target_type", aggfunc=list),
                                           usable=pd.NamedAgg(column="usable", aggfunc=list),
                                           iq_sweep_burst=pd.NamedAgg(column="iq_sweep_burst", aggfunc=list),
                                           doppler_burst=pd.NamedAgg(column="doppler_burst", aggfunc=list),)
    df_tracks['iq_sweep_burst'] = df_tracks['iq_sweep_burst'].apply(lambda x: np.concatenate(x, axis=-1))
    df_tracks['doppler_burst'] = df_tracks['doppler_burst'].apply(lambda x: np.concatenate(x, axis=-1))
    df_tracks['target_type'] = df_tracks['target_type'].apply(np.array)
    df_tracks['usable'] = df_tracks['usable'].apply(np.array)
    return df_tracks.to_dict(orient='index')


def pipeline_trainval(PATH_DATA, config={}):
    """
    arguments:
    ...
    config -- {dict}:
        num_tracks -- {int} -- # of tracks to take from aux dataset
        valratio -- {int} -- Ratio of train/val split
        get_shifts -- {bool} -- Flag to add shifts
        shift_segment -- {int} -- How much to shift tracks to generate new segments
        get_horizontal_flip -- {bool} -- Flag to add horizontal flips
        get_vertical_flip -- {bool} -- Flag to add vertical flips
    """

    ### Default parameter
    num_tracks = config.get('num_tracks', 3)
    val_ratio = config.get('val_ratio', 6)
    shift_segment = config.get('shift_segment', np.arange(1, 31))
    get_shifts = config.get('get_shifts', False)
    get_horizontal_flip = config.get('get_horizontal_flip', False)
    get_vertical_flip = config.get('get_vertical_flip', False)

    ##############################
    #####  LOAD RAW DATA    ######
    ##############################

    experiment_auxiliary = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
    experiment_auxiliary_df = get_data.load_data(experiment_auxiliary, PATH_DATA)
    logger.info(f"experiment_auxiliary:{experiment_auxiliary_df['date_index'].shape}")

    train_aux = get_data.aux_split(experiment_auxiliary_df, numtracks=num_tracks)
    logger.info(f"train_aux:{train_aux['date_index'].shape}")

    train_path = 'MAFAT RADAR Challenge - Training Set V1'
    training_dict = get_data.load_data(train_path, PATH_DATA)

    # Adding segments from the experiment auxiliary set to the training set
    train_dict = get_data.append_dict(training_dict, train_aux)

    logger.info(
        f"training_dict({training_dict['date_index'].shape}) + aux dataset({train_aux['date_index'].shape}) = full train({train_dict['date_index'].shape})")

    full_data = pd.DataFrame.from_dict(train_dict, orient='index').transpose()

    # split Tracks here to only do augmentation on train set
    full_data = get_data.split_train_val_as_df(full_data, ratio=val_ratio)

    logger.info(
        f"train only:{len(full_data[full_data.is_validation == False])}.  val only:{len(full_data[full_data.is_validation == True])}")

    # to free ram space
    del experiment_auxiliary_df
    del train_aux
    del training_dict
    del train_dict

    ###################################
    ##  ADD DATA (VIA AUGMENTATIONS) ##
    ###################################

    full_data['augmentation_info'] = []

    if get_shifts:
        full_data = add_data.generate_shifts(full_data, shift_by=shift_segment)

    if get_vertical_flip:
        full_data = add_data.generate_flips(full_data, type='vertical')

    if get_horizontal_flip:
        full_data = add_data.generate_flips(full_data, type='horizontal')

    ##########################################
    ### TRANSFORMATIONS / DATA ENGINEERING ###
    ##########################################

    ### OPTIONALLY SPLITTING VAL INTO TEST

    ###########################################
    ###             X,y splits              ###
    ###########################################

    train_processed = specto_feat.data_preprocess(train_dict)
    train_x = train_processed['iq_sweep_burst']
    train_x = train_x.reshape(list(train_x.shape) + [1])
    train_y = train_processed['target_type'].astype(int)

    val_processed = specto_feat.data_preprocess(val_dict)
    val_x = val_processed['iq_sweep_burst']
    val_x = val_x.reshape(list(val_x.shape) + [1])
    val_y = val_processed['target_type'].astype(int)

    return train_x, train_y, val_x, val_y



def pipeline_trainval_ram_reduced(PATH_DATA, config = {}):
  '''
  arguments:
      ...
      config -- {dict}:
          num_tracks -- {int} -- # of tracks to take from aux dataset
          valratio -- {int} -- Ratio of train/val split
          get_shifts -- {bool} -- Flag to add shifts
          shift_segment -- {int} -- How much to shift tracks to generate new segments
          get_horizontal_flip -- {bool} -- Flag to add horizontal flips
          get_vertical_flip -- {bool} -- Flag to add vertical flips
          wavelets -- {bool} -- {bool} -- Flag to transform IQ burst into 3-d 7 channels scalograms

  '''

  ### Default parameter
  num_tracks = config.get('num_tracks',3)
  val_ratio = config.get('val_ratio',6)
  shift_segment = config.get('shift_segment',np.arange(1,31))
  get_shifts = config.get('get_shifts',False)
  get_horizontal_flip = config.get('get_horizontal_flip',False)
  get_vertical_flip = config.get('get_vertical_flip',False)
  wavelets = config.get('wavelets',False)

  ##############################
  #####  LOAD RAW DATA    ######
  ##############################

  train_path = 'MAFAT RADAR Challenge - Training Set V1'
  training_dict = get_data.load_data(train_path, PATH_DATA)

  if config.get('include_test_data',False):
    print("adding public testset")
    full_test_path = 'MAFAT RADAR Challenge - FULL Public Test Set V1'
    full_test_df =  get_data.load_data(full_test_path, PATH_DATA)
    training_dict = get_data.append_dict(training_dict, full_test_df)

  if num_tracks==0:
    train_dict = training_dict
  else:
    print("adding auxilary start")
    experiment_auxiliary = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
    experiment_auxiliary_df = get_data.load_data(experiment_auxiliary, PATH_DATA)
    logger.info(f"experiment_auxiliary:{experiment_auxiliary_df['date_index'].shape}")

    train_aux = get_data.aux_split(experiment_auxiliary_df, numtracks= num_tracks)
    logger.info(f"train_aux:{train_aux['date_index'].shape}")

    # Adding segments from the experiment auxiliary set to the training set
    train_dict = get_data.append_dict(training_dict, train_aux)
    logger.info(f"training_dict({training_dict['date_index'].shape}) + aux dataset({train_aux['date_index'].shape}) = full train({train_dict['date_index'].shape})")
    print("adding auxilary end")

    del experiment_auxiliary_df
    del train_aux

  full_data = pd.DataFrame.from_dict(train_dict,orient='index').transpose()

  #split Tracks here to only do augmentation on train set
  full_data = get_data.split_train_val_as_pd(full_data, ratio=val_ratio)

  logger.info(f"train only:{len(full_data[full_data.is_validation == False])}.  val only:{len(full_data[full_data.is_validation == True])}")

  del training_dict
  del train_dict


  ###################################
  ##  ADD DATA (VIA AUGMENTATIONS) ##
  ###################################

  full_data['augmentation_info']=np.empty((len(full_data), 0)).tolist()

  if get_shifts:
    full_data = add_data.db_add_shifts(full_data,shift_by=shift_segment)

  if get_vertical_flip:
    full_data = add_data.db_add_flips(full_data,mode='vertical')

  if get_horizontal_flip:
    full_data = add_data.db_add_flips(full_data,mode='horizontal')


  ##########################################
  ### TRANSFORMATIONS / DATA ENGINEERING ###
  ##########################################

  ### OPTIONALLY SPLITTING VAL INTO TEST
                
  return full_data


if __name__ == "__main__":
                
    print("hello world")
    a, b, c, d = pipeline_trainval('/home/shaul/workspace/GitHub/sota-mafat-radar/data/')