# import sys
# sys.path.append('/home/shaul/workspace/GitHub/sota-mafat-radar')

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

def pipeline_trainval(PATH_DATA, config = {}):
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

  '''

  ### Default parameter
  num_tracks = config.get('num_tracks',3)
  val_ratio = config.get('val_ratio',6)
  shift_segment = config.get('shift_segment',np.arange(1,31))
  get_shifts = config.get('get_shifts',False)
  get_horizontal_flip = config.get('get_horizontal_flip',False)
  get_vertical_flip = config.get('get_vertical_flip',False)

  #TODO: Add logger for how much data we have (due to augmentations, etc.)

  ##############################
  #####  LOAD RAW DATA    ######
  ##############################

  experiment_auxiliary = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
  experiment_auxiliary_df = get_data.load_data(experiment_auxiliary, PATH_DATA)
  logger.info(f"experiment_auxiliary:{experiment_auxiliary_df['date_index'].shape}")

  train_aux = get_data.aux_split(experiment_auxiliary_df, numtracks= num_tracks)
  logger.info(f"train_aux:{train_aux['date_index'].shape}")

  train_path = 'MAFAT RADAR Challenge - Training Set V1'
  training_dict = get_data.load_data(train_path, PATH_DATA)

  # Adding segments from the experiment auxiliary set to the training set
  train_dict = get_data.append_dict(training_dict, train_aux)

  # to free ram space
  del experiment_auxiliary_df 

  logger.info(f"training_dict({training_dict['date_index'].shape}) + aux dataset({train_aux['date_index'].shape}) = full train({train_dict['date_index'].shape})")

  #split Tracks here to only do augmentation on Train set
  train_dict, val_dict = get_data.split_train_val_as_df(train_dict,ratio= val_ratio)

  logger.info(f"train only:{train_dict['iq_sweep_burst'].shape}.  val only:{val_dict['iq_sweep_burst'].shape}")


  ###################################
  ##  ADD DATA (VIA AUGMENTATIONS) ##
  ###################################

  # Splitting the tracks into new segments

  if get_shifts:


    logger.info(f"shifts. ram1:{psutil.virtual_memory().percent}")
    train_df = train_dict.copy()
    logger.info(f"shifts. ram2:{psutil.virtual_memory().percent}")

    del train_df['doppler_burst']
    del train_df['iq_sweep_burst']
    train_df = pd.DataFrame(train_df)

    new_segments_results = add_data.generate_shifts(train_df,train_dict,shift_by=shift_segment)
    shifted_ds_dict = {k: [dic[k] for dic in new_segments_results] for k in new_segments_results[0]}
    train_dict = get_data.append_dict(train_dict, shifted_ds_dict)

    logger.handlers[0].flush()
    logger.info(f"train only (after adding shifts):{train_dict['iq_sweep_burst'].shape[0]}.  val only:{val_dict['iq_sweep_burst'].shape[0]}")

  #train_og = train_dict.copy()

  if get_vertical_flip:
    add_vertical = {'iq_sweep_burst':None,'doppler_burst':None}
    add_vertical['iq_sweep_burst'] = augmentations.vertical_flip(add_vertical['iq_sweep_burst'])
    add_vertical['doppler_burst'] = 128 - add_vertical['doppler_burst']
    train_dict = get_data.append_dict(train_dict, add_vertical)

  if get_horizontal_flip:
    add_horizontal = {'iq_sweep_burst':None,'doppler_burst':None}
    add_horizontal['iq_sweep_burst'] = augmentations.horizontal_flip(add_horizontal['iq_sweep_burst'])
    add_horizontal['doppler_burst'] = np.flip(add_horizontal['doppler_burst'],axis=1)
    train_dict = get_data.append_dict(train_dict, add_horizontal)

  ##########################################
  ### TRANSFORMATIONS / DATA ENGINEERING ###
  ##########################################

  ### OPTIONALLY SPLITTING VAL INTO TEST

  ###########################################
  ###             X,y splits              ###
  ###########################################

  train_processed = specto_feat.data_preprocess(train_dict)
  train_x = train_processed['iq_sweep_burst']
  train_x = train_x.reshape(list(train_x.shape)+[1])
  train_y = train_processed['target_type'].astype(int)

  val_processed = specto_feat.data_preprocess(val_dict)
  val_x =  val_processed['iq_sweep_burst']
  val_x = val_x.reshape(list(val_x.shape)+[1])
  val_y = val_processed['target_type'].astype(int)

  return train_x, train_y, val_x, val_y

if __name__ == "__main__":
  print("hello world")
  a,b,c,d = pipeline_trainval('/home/shaul/workspace/GitHub/sota-mafat-radar/data/')

