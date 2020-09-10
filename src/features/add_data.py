import numpy as np
import os
import pickle
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import itertools
# from matplotlib.colors import LinearSegmentedColormap
# import configparser
# import matplotlib.patches as patches
# import math
# from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
# from sklearn.manifold import TSNE
# from tensorflow.keras.models import Model


def concatenate_track(data, track_id, snr_plot='both'):
    """Concatenate segments with same track id
    
    Arguments:
    data -- {dictionary} -- python dictionary of python numpy arrays
    track_id -- {int} -- the track_id number of the wanted segments
    snr_plot -- {str} -- If track has both high and low SNR signals which SNR to plot (Default = 'both')
      The valid values are: 'HighSNR', 'LowSNR' or 'both'
      
    Returns:
    Concatenated I/Q matrix and concatenated doppler burst vector
    """
    track_indices = np.where(data['track_id'] == track_id)
    iq_list = []
    dopller_list = []

    if (snr_plot != 'both') and (not has_single_snr_type(data, track_id, False)):
      track_indices = np.where((data['track_id'] == track_id) & (data['snr_type'] == snr_plot))
    
    for i in track_indices:
        iq_list.append(data['iq_sweep_burst'][i])
        dopller_list.append(data['doppler_burst'][i])
      
    iq_matrix = np.concatenate(np.concatenate(iq_list, axis=1),axis=1)
    doppler_vector = np.concatenate(np.concatenate(dopller_list, axis=0),axis=0)
    
    return iq_matrix, doppler_vector
    

def generate_shifts(data_df,data,shift_by=None):
  """
  generate shifts from the data. important: pay attention if preprocessing has already been done on the data!!
  preprocess 'merges' the burst into the iq values.
  Arguments:
    data_df -- {dataframe} -- parameters for each segment (geo type+id, snr etc)
    data -- {ndarray} -- the data set (only iq and burst)
    shift_by -- (int/array) Validation / Test (used in syntehtic test)
  Returns:
    list of dictionary. each item in list holds the parameter of a new shifted segments + iq + burst
  """  
  new_segments_results = []

  all_track_ids = data_df.track_id.unique()

  
  if type(shift_by) is not list:
    shift_by_list = [shift_by]
  else:
    shift_by_list = shift_by

  for shift_by_i in shift_by_list:

    for track_id_t in all_track_ids:

      segment_idxs = list(data_df[data_df.track_id==track_id_t].index)
      segment_idxs = [(x,y) for x,y in zip(segment_idxs, segment_idxs[1:])]

      iq,burst = concatenate_track(data, track_id_t, snr_plot='both')

      x_ind = -32
      for seg_id in segment_idxs:
          x_ind = x_ind +32
          #print(data.iloc[seg_id])

          columns = ['geolocation_type','geolocation_id','sensor_id','snr_type','date_index','target_type']
          for col in columns:
            if data_df.iloc[seg_id[0]][col] != data_df.iloc[seg_id[1]][col]:
              #print(f"{seg_id[0]},{seg_id[1]}: diff {col}. skip")
              continue

          # if data_df.iloc[seg_id[0]].is_validation or data_df.iloc[seg_id[1]].is_validation:
          #   #print(f"{seg_id[0]},{seg_id[1]}: is_validation. skip")
          #   continue

          new_seg_start = x_ind+shift_by_i

          #print(f"new seg: {new_seg_start}-{new_seg_start+32}")
          new_segments_results.append({
              'segment_id': 100000 + data_df.iloc[seg_id[0]].segment_id,
              'track_id': data_df.iloc[seg_id[0]].track_id,
              'geolocation_type': data_df.iloc[seg_id[0]].geolocation_type,
              'geolocation_id': data_df.iloc[seg_id[0]].geolocation_id,
              'sensor_id': data_df.iloc[seg_id[0]].sensor_id,
              'snr_type': data_df.iloc[seg_id[0]].snr_type,
              'date_index': data_df.iloc[seg_id[0]].date_index,
              'target_type': data_df.iloc[seg_id[0]].target_type,
              'is_validation': False,
              'iq_sweep_burst': iq[:,new_seg_start:new_seg_start+32],
              'doppler_burst': burst[new_seg_start:new_seg_start+32], 
              'shift': shift_by_i
          })

  return new_segments_results
