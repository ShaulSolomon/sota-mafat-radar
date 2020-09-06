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
from tensorflow.keras.models import Model

from features.specto_feat import calculate_spectrogram

def plot_spectrogram(iq_burst, doppler_burst, color_map_name='parula',
                    color_map_path=None, save_path=None, flip=True, return_spec=False, 
                    figsize=None, label=None, ax=None, title=None,val_overlay=None,
                    theta=None):
    """
    Plots spectrogram of 'iq_sweep_burst'.

    Arguments:

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
    figsize -- {tuple} -- plot the spectrogram with the given figsize (Default = None)
    label -- {str} -- String to pass as plot title (Default = None)
    ax -- {plt ax} -- plt ax object. can be used to show the result in subplots
    title -- title for the plot
    val_overlay -- (list) draw a rectangle around validation segments, red for fail, green for success
    theta -- {float} degrees to rotate spectogram by, in radians (Default = None)
  Returns:
    Spectrogram data if return_spec is True
    """
    if color_map_path is not None:
        cm_data = np.load(color_map_path)
        color_map = LinearSegmentedColormap.from_list(color_map_name, cm_data)
    elif color_map_name == 'parula':
        print("Error: when 'parula' color map is used, color_map_path should be provided.")
        print("Switching color map to 'viridis'.")
        color_map = LinearSegmentedColormap.from_list(color_map_name, spectrogram_cmap)
    else:
        color_map = plt.get_cmap(color_map_name)

    iq = calculate_spectrogram(iq_burst, flip=flip)
#     if theta:
#         iq = rotate_spectogram(iq, theta)
    if return_spec:
        return iq

    plt_o = plt
    if ax is not None: 
        plt_o = ax

    if figsize is not None:
        plt_o.rcParams["figure.figsize"] = figsize


    if doppler_burst is not None:
        pixel_shift = 0.5
        if flip:
            plt_o.plot(pixel_shift + np.arange(len(doppler_burst)),
                       pixel_shift + (len(iq) - doppler_burst), '.w')
        else:
            plt_o.plot(pixel_shift + np.arange(len(doppler_burst)), pixel_shift + doppler_burst, '.w')

    ax1 = plt.gca()
    if val_overlay is not None:
        for i,seg in enumerate(val_overlay):
            if seg is None: 
                continue
        overlay_color = 'g' if seg==True else 'r'
        x_pos = i*32
        rect = patches.Rectangle((x_pos,0),31,127,linewidth=2,edgecolor=overlay_color,facecolor='none')
        ax1.add_patch(rect)

    # plt_o.imshow(iq, cmap=color_map)

    if save_path is not None:
        plt_o.imsave(save_path, iq, cmap=color_map)

    if title is not None:
        if ax is None:
            plt_o.title(title)
        else:
            plt_o.set_title(title)

    if ax is None: 
        if isinstance(label, str): plt_o.title(label)
        # plt_o.show()
        # plt_o.clf()
