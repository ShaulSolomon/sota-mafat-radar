from src.utils import experiment_utils as utils
from termcolor import colored
import numpy as np
import pandas as pd

# The function append_dict is for concatenating the training set 
# with the Auxiliary data set segments

def append_dict(dict1, dict2):
  for key in dict1:
    dict1[key] = np.concatenate([dict1[key], dict2[key]], axis=0)
  return dict1

def classic_trainvalSNR(PATH_DATA):

    # Set and test path to competition data files with snr_type as additional feature
    
    try:
        file_path = 'MAFAT RADAR Challenge - Training Set V1.csv'
        with open(f'{PATH_DATA}/{file_path}') as f:
            f.readlines()
        print(colored('Everything is setup correctly', color='green'))
    except:
        print(colored('Please mount drive and set competition_path correctly',
                        color='red'))

    # Loading and preparing the data
    # Loading Auxiliary Experiment set - can take a few minutes
    experiment_auxiliary = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
    experiment_auxiliary_df = utils.load_data(experiment_auxiliary, PATH_DATA)

    train_aux = utils.aux_split(experiment_auxiliary_df)

    # Training set
    train_path = 'MAFAT RADAR Challenge - Training Set V1'
    training_df = utils.load_data(train_path, PATH_DATA)

    # Adding segments from the experiment auxiliary set to the training set
    train_df = append_dict(training_df, train_aux)


    # Preprocessing and split the data to training and validation
    train_df = utils.data_preprocess(train_df.copy())
    
    
    train_x, train_y, val_x, val_y, idx = utils.split_train_val(train_df)

    train_snr = np.array(pd.Series(train_df['snr_type'][np.logical_not(idx)]).apply(lambda x: 0 if x=="HighSNR" else 1).astype(int))
    val_snr = np.array(pd.Series(train_df['snr_type'][idx]).apply(lambda x: 0 if x=="HighSNR" else 1).astype(int))
    val_y =  val_y.astype(int)
    train_y =train_y.astype(int)
    train_x = train_x.reshape(list(train_x.shape)+[1])
    val_x = val_x.reshape(list(val_x.shape)+[1])

    return train_x,train_snr, train_y, val_x,val_snr, val_y



def classic_trainval(PATH_DATA):

    # Set and test path to competition data files
    try:
        file_path = 'MAFAT RADAR Challenge - Training Set V1.csv'
        with open(f'{PATH_DATA}/{file_path}') as f:
            f.readlines()
        print(colored('Everything is setup correctly', color='green'))
    except:
        print(colored('Please mount drive and set competition_path correctly',
                        color='red'))

    # Loading and preparing the data
    # Loading Auxiliary Experiment set - can take a few minutes
    experiment_auxiliary = 'MAFAT RADAR Challenge - Auxiliary Experiment Set V2'
    experiment_auxiliary_df = utils.load_data(experiment_auxiliary, PATH_DATA)

    train_aux = utils.aux_split(experiment_auxiliary_df)

    # Training set
    train_path = 'MAFAT RADAR Challenge - Training Set V1'
    training_df = utils.load_data(train_path, PATH_DATA)

    # Adding segments from the experiment auxiliary set to the training set
    train_df = append_dict(training_df, train_aux)

    # Preprocessing and split the data to training and validation
    train_df = utils.data_preprocess(train_df.copy())
    train_x, train_y, val_x, val_y, _ = utils.split_train_val(train_df)

    val_y =  val_y.astype(int)
    train_y =train_y.astype(int)
    train_x = train_x.reshape(list(train_x.shape)+[1])
    val_x = val_x.reshape(list(val_x.shape)+[1])

    return train_x, train_y, val_x, val_y