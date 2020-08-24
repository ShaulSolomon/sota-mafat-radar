import numpy as np
# import os
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import itertools
# from matplotlib.colors import LinearSegmentedColormap
# import configparser
# import matplotlib.patches as patches
# import math
# from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
# from sklearn.manifold import TSNE


def get_track_id(data, segment_id):
  """
  Get track id from segment id.

  Arguments:
    data -- {dictionary} -- python dictionary of python numpy arrays
    segment_id -- {int} -- segment id of a track

  Returns:
    Track id
  """
  segment_index = np.where(data['segment_id'] == segment_id)
  return data['track_id'][segment_index][0]


def has_single_snr_type(data, id, is_segment):
  """
  Check if a track has a single SNR type or both High and Low SNR.

  Arguments:
    data -- {dictionary} -- python dictionary of python numpy arrays
    id -- {int} -- segment or track id, based on is_segment
    is_segment -- {bool} -- If true then id is segment, otherwise id is track

  Returns:
    True if track has a High or Low SNR but not both
  """
  if is_segment:
    id = get_track_id(data, id)
  return np.all(data['snr_type'][np.where(data['track_id'] == id)] == data['snr_type']\
                [np.where(data['track_id'] == id)][0], axis = 0)

def get_label(data, segment_id=None, track_id=None):
    """
    Arguments:
    data -- {dictionary} -- python dictionary of python numpy arrays
    segment_id -- {int} -- segment id of a track
    track_id -- {int} -- the track_id number of the wanted segment
      
    Returns:
    String with label for track
    """
    labels = data.get('target_type', None)
    if labels is None:
        print('Dataset does not have labels')
        return labels
    if (segment_id == None) and (track_id == None):
        raise ValueError("You must pass segment id or track id")
    elif (segment_id != None) and (track_id != None):
        raise ValueError("You must pass segment id or track id, you can't pass both.")
    elif (segment_id != None) and (track_id == None):
        segment_index = np.where(data['segment_id'] == segment_id)
        label = np.unique(labels[segment_index])
    else:
        track_indices = np.where(data['track_id'] == track_id)
        label = np.unique(labels[track_indices])
    if len(label) == 1: 
        return label.item()
    else:
        print(f"{len(label)} labels in segment: Labels: {label}")
        return None