import sys
sys.path.append('/home/shaul/workspace/GitHub/sota-mafat-radar')

import numpy as np
import os
import pandas as pd
from termcolor import colored
import configparser
import pickle
from src.data import get_data

def pipeline_trainval(PATH_DATA, config = {}):
    '''

    arguments:
        ...
        config -- {dict}:
            num_tracks -- {int} -- # of tracks to take from aux dataset
            valratio -- {int} -- Ratio of train/val split

    '''

    ### Default parameter
    num_tracks = config.get('num_tracks',3)
    val_ratio = config.get('val_ratio',6)

    #TODO: Add logger for how much data we have (due to augmentations, etc.)

    ### LOAD RAW DATA

    experiment_auxiliary = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
    experiment_auxiliary_df = get_data.load_data(experiment_auxiliary, PATH_DATA)

    train_aux = get_data.aux_split(experiment_auxiliary_df, numtracks= num_tracks)

    train_path = 'MAFAT RADAR Challenge - Training Set V1'
    training_dict = get_data.load_data(train_path, PATH_DATA)

    # Adding segments from the experiment auxiliary set to the training set
    train_dict = get_data.append_dict(training_dict, train_aux)

    #split Tracks here to only do augmentation on Train set
    train_dict, val_dict = get_data.split_train_val_as_df(train_dict,ratio= val_ratio)

    ### ADD DATA (VIA AUGMENTATIONS)

    # Splitting the tracks into new segments




    ### TRANSFORMATIONS / DATA ENGINEERING

    ### OPTIONALLY SPLITTING VAL INTO TEST





  # Preprocessing and split the data to training and validation


#   train_df = specto_feat.data_preprocess(train_df.copy())
#   train_x, train_y, val_x, val_y, _ = split_train_val(train_df)

#   val_y =  val_y.astype(int)
#   train_y =train_y.astype(int)
#   train_x = train_x.reshape(list(train_x.shape)+[1])
#   val_x = val_x.reshape(list(val_x.shape)+[1])

#   return train_x, train_y, val_x, val_y


if __name__ == "__main__":
  print("hello world")
  a,b,c,d = pipeline_trainval('/home/shaul/workspace/GitHub/sota-mafat-radar/data/')




'''

def pipepine(*args):

def pipeline(dict_params)


'''