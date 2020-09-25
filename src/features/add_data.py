import numpy as np
import os
import pickle
import pandas as pd
import psutil
import sys
from tqdm import tqdm_notebook as tqdm
from src.data.feat_data import has_single_snr_type
import logging
logger = logging.getLogger()


def concatenate_track(data, track_id, snr_plot='both', return_indices=False):

    """Concatenate segments with same track id

    Arguments:
    data -- {dictionary / dataframe} -- python dictionary of python numpy arrays
    track_id -- {int} -- the track_id number of the wanted segments
    snr_plot -- {str} -- If track has both high and low SNR signals which SNR to plot (Default = 'both')
      The valid values are: 'HighSNR', 'LowSNR' or 'both'

    Returns:
    Concatenated I/Q matrix and concatenated doppler burst vector
    """
    iq_list = []
    doppler_list = []

    if isinstance(data, dict):
        track_indices = np.where(data['track_id'] == track_id)
        if (snr_plot != 'both') and (not has_single_snr_type(data, track_id, False)):
            track_indices = np.where((data['track_id'] == track_id) & (data['snr_type'] == snr_plot))
        track_indices = list(track_indices[0])

    elif isinstance(data, pd.DataFrame):
        if (snr_plot != 'both'):
            track_indices = list(data[(data['track_id'] == track_id) & (data['snr_type'] == snr_plot)].index)
        else:
            track_indices = list(data[data['track_id'] == track_id].index)

    # print(f"track_id:{track_id},snr_plot:{snr_plot},track_indices:{track_indices}")

    else:
      if (snr_plot != 'both'):
        track_indices = list(data[(data['track_id'] == track_id) & (data['snr_type'] == snr_plot)].index)
      else:
        track_indices = list(data[data['track_id'] == track_id].index)

    #print(f"track_id:{track_id},snr_plot:{snr_plot},track_indices:{track_indices}")

    for i in track_indices:
        if data['iq_sweep_burst'][i] is not None:
            iq_list.append(data['iq_sweep_burst'][i])
            doppler_list.append(data['doppler_burst'][i])

    # print(f"iq_list:{iq_list}. shape:{iq_list[0].shape}. len:{len(iq_list)}")

    iq_matrix = np.concatenate(iq_list, axis=-1)
    doppler_vector = np.concatenate(doppler_list, axis=-1)
    if return_indices:
        return iq_matrix, doppler_vector, track_indices
    else:
        return iq_matrix, doppler_vector

def shifts_from_track(data, shift_by=None):
    """
        generate shifts from one track.
        Arguments:
            data -- {dataframe} -- dataframe N number of tracks, including parameters for each segment (geo type+id, snr etc)
            shift_by -- (int/array) Validation / Test (used in syntehtic test)
        Returns:
            dataframe same as input, with appended rows of the shifts appended at the end. column 'augmentation_info' has the
            instrunctions on how to create the augmentation
      """
    pass
    shift_by_list = np.arange(0, 31, shift_by).tolist()

def generate_shifts(data_df,data, shift_by: int = 1):
  """
  generate shifts from the data.
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


def db_add_shifts(full_data,shift_by=None):
  """
  generate shifts from the data.
  Arguments:
    full_data -- {dataframe} -- dataframe with all data, including  parameters for each segment (geo type+id, snr etc)
    shift_by -- (int/array) Validation / Test (used in syntehtic test)
  Returns:
    dataframe same as input, with appended rows of the shifts appended at the end. column 'augmentation_info' has the
    instrunctions on how to create the augmentation
  """
  new_segments_results = []
  count = 0

  all_track_ids = full_data.track_id.unique()
  data = full_data.copy() # copy the data to prevent duplicates that when adding additional rows to full_data dataframe

  if type(shift_by) is not list:
    shift_by_list = [shift_by]
  else:
    shift_by_list = shift_by

  for shift_by_i in shift_by_list:

    logger.info(f"shift:{shift_by_i}")
    print(f"shift:{shift_by_i}")

    for track_id_t in tqdm(all_track_ids):

      logger.info(f"track:{track_id_t} | ram:{psutil.virtual_memory().percent},{sys.getsizeof(new_segments_results)}")
      #print(f"track:{track_id_t} | ram:{psutil.virtual_memory().percent},{sys.getsizeof(new_segments_results)}")

      segment_idxs = list(data[data.track_id==track_id_t].index)
      segment_idxs = [(x,y) for x,y in zip(segment_idxs, segment_idxs[1:])]

      iq,burst = concatenate_track(data, track_id_t, snr_plot='both')

      x_ind = -32
      for seg_id in segment_idxs:

          logger.info(f"seg_id:{seg_id}")

          x_ind = x_ind +32
          #print(data.iloc[seg_id])

          columns = ['geolocation_type','geolocation_id','sensor_id','snr_type','date_index','target_type']
          ok_to_add = True
          for col in columns:
            if data.iloc[seg_id[0]][col] != data.iloc[seg_id[1]][col]:
              #print(f"{seg_id[0]},{seg_id[1]}: diff {col}. skip")
              ok_to_add = False

            if data.iloc[seg_id[0]].is_validation or data.iloc[seg_id[1]].is_validation:
              #print(f"{seg_id[0]},{seg_id[1]}: is_validation. skip")
              ok_to_add = False

          if ok_to_add:
            new_seg_start = x_ind+shift_by_i
            count = count+1

            new_segments_results = {
                'segment_id': 100000 + data.iloc[seg_id[0]].segment_id,
                'track_id': data.iloc[seg_id[0]].track_id,
                'geolocation_type': data.iloc[seg_id[0]].geolocation_type,
                'geolocation_id': data.iloc[seg_id[0]].geolocation_id,
                'sensor_id': data.iloc[seg_id[0]].sensor_id,
                'snr_type': data.iloc[seg_id[0]].snr_type,
                'date_index': data.iloc[seg_id[0]].date_index,
                'target_type': data.iloc[seg_id[0]].target_type,
                'is_validation': False,
                'augmentation_info': [{
                  'type':'shift',
                  'shift': shift_by_i,
                  'from_segments': seg_id
                }],
                'iq_sweep_burst': None,
                'doppler_burst': None,
            }

            full_data = full_data.append(new_segments_results, ignore_index=True)
            logger.info(f"cnt:{count}. new seg:{new_seg_start}-{new_seg_start+32}. track:{track_id_t}. len:{len(full_data)}")

          logger.handlers[0].flush()

  return full_data


def add_flip_augmentation(x,mydict):
  mydict['from_segment'] = x.segment_id
  x.augmentation_info = x.augmentation_info + [mydict]
  return x.augmentation_info


def db_add_flips(full_data,mode=None):
  """
  generate shifts from the data.
  Arguments:
    full_data -- {dataframe} -- dataframe with all data, including  parameters for each segment (geo type+id, snr etc)
    mode -- {string} -- 'horizontal' / 'vertical'
  Returns:
    dataframe same as input, with appended rows of the shifts appended at the end. column 'augmentation_info' has the
    instrunctions on how to create the augmentation
  """

  df = full_data.copy()
  df = df.drop(df[df.is_validation == True].index)

  seg_id_start  = max(df['segment_id'])

  add_flips_dict = {
        'type':'flip',
        'mode': mode,
  }

  df.augmentation_info = df.apply(add_flip_augmentation,mydict=add_flips_dict, axis=1)
  df.segment_id = df.segment_id + seg_id_start +1

  return pd.concat([full_data, df], ignore_index=True)