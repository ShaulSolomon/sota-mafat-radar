import numpy as np
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from matplotlib.colors import LinearSegmentedColormap
import configparser
import matplotlib.patches as patches
import math
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.manifold import TSNE
import pywt
import tqdm


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


def calculate_spectrogram(iq_burst, axis=0, flip=True):
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
    if flip:
        iq = np.flip(iq, axis=0)

    return iq

def spectrogram(data, segment_id=None, plot_track=False, track_id=None, snr_plot='both',
                color_map_name='parula', color_map_path=None, save_path=None, flip=True,
                return_spec=False, give_label=True, title=None,val_overlay=None):
    """
  Plots spectrogram of a track or of a single segment I/Q matrix ('iq_sweep_burst').
  If segment_id is passed than plots spectrogram for the specific segment,
  unless plot_track=='True' and than plots the entire track of the segment.
  If track_id is passed than plots spectrogram for the entire track.
  In case that the Track has two SNR Types asks the user to choose HighSNR or LowSNR or ignore.
  If color map is 'parula' must pass color_map_path.        

  Arguments:
    data -- {dictionary} -- python dictionary of python numpy arrays
    segment_id -- {int} -- the segment_id number of the wanted segment
    track_id -- {int} -- the segment_id number of the wanted segment
    snr_plot -- {str} -- If track has both high and low SNR signals which SNR to plot (Default = 'both')
      The valid values are: 'HighSNR', 'LowSNR' or 'both'
    iq_burst -- {ndarray} -- 'iq_sweep_burst' array
    doppler_burst -- {ndarray} -- 'doppler_burst' array (center of mass)
      if is 0 (or zero array) then not plotted
    color_map_name -- {str}  -- name of color map to be used (Default = 'parula')
      if 'parula' is set then color_map_path must be provided
    color_map_path -- {str} -- path to color_map file (Default=None)
      if None then default color map is used
    save_path -- {str} -- path to save image (Default = None)
      if None then saving is not performed
    flip -- {bool} -- flip the spectrogram to match Matlab spectrogram (Default = True)
    return_spec -- {bool} -- if True, returns spectrogram data and skips plotting and saving
    give_label -- {bool} -- if True, adds label ("human, animal") as plot title (Default = True)
    title -- title for the plot
    val_overlay -- (list) draw a rectangle around validation segments, red for fail, green for success  
  Returns:
    Spectrogram data if return_spec is True
    """
    label = None
    if give_label:
        label = get_label(data, segment_id=segment_id, track_id=track_id)
    if (segment_id == None) and (track_id == None):
        raise ValueError("You must pass segment id or track id")
    elif (segment_id != None) and (track_id != None):
        raise ValueError("You must pass segment id or track id, you can't pass both.",
    "\nIf you want to plot the entire track of a segment by passig only the segment_id than set 'plot_track'=True")
    elif (segment_id != None) and (track_id == None):
        segment_index = np.where(data['segment_id'] == segment_id)
        if not plot_track:
            iq_matrix = data['iq_sweep_burst'][segment_index]
            iq_matrix = iq_matrix.reshape(iq_matrix.shape[1], -1)
            doppler_vector = data['doppler_burst'][segment_index]
            doppler_vector = doppler_vector.reshape(doppler_vector.shape[1])
            plot_spectrogram(iq_burst=iq_matrix, doppler_burst=doppler_vector, color_map_name=color_map_name,
                             color_map_path=color_map_path, save_path=save_path, flip=flip, return_spec=return_spec,
                             label=label, title=title, val_overlay=val_overlay)
        else:
            '''plot_track=True than plots all track by segment_id'''
            track_id = data['track_id'][segment_index]
            spectrogram(data, segment_id=None, plot_track=False, track_id=track_id,
                             snr_plot=snr_plot, color_map_name=color_map_name, 
                             color_map_path=color_map_path, save_path=save_path, flip=flip, 
                             return_spec=return_spec,title=title, val_overlay=val_overlay)
    else:
        ''' track_id is passed, plotting the entire track '''
    
        iq_matrix, doppler_vector = concatenate_track(data, track_id, snr_plot) 

        plot_spectrogram(iq_burst=iq_matrix, doppler_burst=doppler_vector, 
                         color_map_name=color_map_name, color_map_path=color_map_path, 
                         save_path=save_path, flip=flip, return_spec=return_spec,
                         label=label, title=title, val_overlay=val_overlay)
    

def fft(iq, axis=0):
  """
  Computes the log of discrete Fourier Transform (DFT).
     
  Arguments:
    iq_burst -- {ndarray} -- 'iq_sweep_burst' array
    axis -- {int} -- axis to perform fft in (Default = 0)

  Returns:
    log of DFT on iq_burst array
  """
  iq = np.log(np.abs(np.fft.fft(hann(iq), axis=axis)))
  return iq


def max_value_on_doppler(iq, doppler_burst):
  """
  Set max value on I/Q matrix using doppler burst vector. 
     
  Arguments:
    iq_burst -- {ndarray} -- 'iq_sweep_burst' array
    doppler_burst -- {ndarray} -- 'doppler_burst' array (center of mass)
               
  Returns:
    I/Q matrix with the max value instead of the original values
    The doppler burst marks the matrix values to change by max value
  """
  iq_max_value = np.max(iq)
  for i in range(iq.shape[1]):
    if doppler_burst[i]>=len(iq):
       continue
    iq[doppler_burst[i], i] = iq_max_value
  return iq


def normalize(iq):
  """
  Calculates normalized values for iq_sweep_burst matrix:
  (vlaue-mean)/std.
  """
  m = iq.mean()
  s = iq.std()
  return (iq-m)/s


def data_preprocess(data, df_type = 'spectrogram' , flip = True, kernel = 'cgau1'):
  """
  Preforms data preprocessing.
  Change target_type lables from string to integer:
  'human'  --> 1
  'animal' --> 0

  Arguments:
    data -- {ndarray} -- the data set
    df_type -- {bool} -- type of processing procedure for the data, either wavelets scalogram, or spectrogram 
    flip -- {bool} -- flip argument for scalogram
    kernel -- {str} -- mother wavelet type for scalogram

  Returns:
    processed data (max values by doppler burst, DFT, normalization, scalogram)
  """
  X=[]
  if df_type == 'scalogram':
    pbar = tqdm.tqdm(total = len(data['iq_sweep_burst']), position = 0, leave = True)
    for i in range(len(data['iq_sweep_burst'])):
      X.append(calculate_scalogram(data['iq_sweep_burst'][i], flip = flip, transformation= kernel))
      pbar.update()
    pbar.close()
    data['scalogram'] = np.array(X)
  else:
    for i in range(len(data['iq_sweep_burst'])):
      iq = fft(data['iq_sweep_burst'][i])
      iq = max_value_on_doppler(iq,data['doppler_burst'][i])
      iq = normalize(iq)
      X.append(iq)
    data['iq_sweep_burst'] = np.array(X)

  if 'target_type' in data:
    data['target_type'][data['target_type'] == 'animal'] = 0
    data['target_type'][data['target_type'] == 'human'] = 1
  return data


def calculate_scalogram(iq_matrix, flip=True, transformation = 'cgau1'):
    '''
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
    '''

    
    scalograms = []
    #analyze each column (time-point) seperatly.
    iq_matrix = normalize(iq_matrix)
    for j in range(iq_matrix.shape[1]):
        # preform hann smoothing on a column - results in a singal j-2 sized column
        # preform py.cwt transformation, returns coefficients and frequencies
        
        coef, freqs=pywt.cwt(hann(iq_matrix[:, j][:, np.newaxis]), np.arange(1,9), transformation)
        # coefficient matrix returns as a (num_scalers-1, j-2 , 1) array, transform it into a 2-d array
        # note: the feqs is a returned value from cwt function that we don't use

        if flip:
            coef = np.flip(coef, axis=0)
        # log normalization of the data
        coef=np.log(np.abs(coef))
        # first column correspond to the scales, rest is the coefficients
        coef=coef[:, :,0]
        
        scalograms.append(coef)

    stacked_scalogram = np.stack(scalograms)
    stacked_scalogram = np.maximum(np.median(stacked_scalogram) - 1., stacked_scalogram)
    stacked_scalogram = np.transpose(stacked_scalogram,(2,0,1))
    return stacked_scalogram


def calculate_scalogram_2d(iq_matrix, flip=True, transformation = 'cgau1'):
  '''
  calculate a scalogram matrix that preforms a continues wavelet transformation on the data.
  takes all the normalized scales and return a singular plane, useful for plotting

  Arguments:
      iq_matrix (array-like): array of complex signal data, rows represent spatial location, columns time
      flip (bool): optional argument for flipping the row order of the matrix.
      transformation (string): name of wavelet signal to use as mother signal. default to gaussian kernel
  return:
      scalogram: array like transformation that correspond with correlation of different frequency wavelets at different time-points. 
  
  1. select each column of the IQ matrix
  2. apply hann-window smoothing
  3. preform Continues Wavelet Transformation (data, array of psooible scale values, type of transformation)
  '''

  scalograms = []
  #analyze each column (time-point) seperatly
  for j in range(iq_matrix.shape[1]):
      # preform hann smoothing on a column - results in a singal j-2 sized column
      # preform py.cwt transformation, returns coefficients and frequencies
      coef, freqs=pywt.cwt(hann(iq_matrix[:, j][:, np.newaxis]), np.arange(1,8), transformation)
      # coefficient matrix returns as a (num_scalers-1, j-2 , 1) array, transform it into a 2-d array
      #coef = coef[:,:,0]
      if flip:
          coef = np.flip(coef, axis=0)
      # log normalization of the data
      coef=np.log(np.abs(coef))
      # first column correspond to the scales, rest is the coefficients
      coef=coef[:, 1:,0]
      coef=coef.T
      scalograms.append(coef)
  stacked_scalogram = np.mean(np.array(scalograms),axis = 2).T
  stacked_scalogram = np.maximum(np.median(stacked_scalogram) - 1., stacked_scalogram)
  return stacked_scalogram