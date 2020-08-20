#!/usr/bin/env python3

import os
import argparse
import shutil

import numpy as np

import utils.experiment_utils as eutils


def get_args():
   parser = argparse.ArgumentParser(description = 'Study a dataset')
   parser.add_argument('-d', '--dataset_path', type=str, required=True, help='dir with pkl & csv')
   parser.add_argument('-o', '--output_path', type=str, required=True, help='dir with pngs')
   parser.add_argument('-f', '--few', type=int, default=10, help='few is how much?')
   args = parser.parse_args()
   assert os.path.isdir(args.dataset_path), args.dataset_path
   assert args.few > 0 and args.few < 1000, args.few
   if os.path.isdir(args.output_path):
       shutil.rmtree(args.output_path)
   os.mkdir(args.output_path)
   return args


def visualize_a_few(dataset_path, output_path, few):
    data = eutils.load_data('train', dataset_path)
    half = few // 2
    human_indices = np.where(data['target_type'] == 'human')[0]
    all_indices = set(range(len(data['segment_id'])))
    animal_indices = list(all_indices - set(human_indices))

    chosen_human_sids = np.random.choice(human_indices, size=half, replace=False)
    chosen_animal_sids = np.random.choice(animal_indices, size=half, replace=False)
    chosen_sids = list(chosen_human_sids) + list(chosen_animal_sids)
    for sid in chosen_sids:
        print('Sample ID ', sid)
        print('iq_sweep_burst:', data['iq_sweep_burst'][sid].shape)
        print('iq_sweep_burst:', data['iq_sweep_burst'][sid][0,0])
        print('doppler_burst:', data['doppler_burst'][sid].shape)
        print('doppler_burst:', data['doppler_burst'][sid][0])
        print('target_type:', data['target_type'][sid])
        print('snr_type: {}\n'.format(data['snr_type'][sid]))

        print('len:', len(data['iq_sweep_burst'][sid]))
        spectrogram = eutils.calculate_spectrogram(data['iq_sweep_burst'][sid])
        eutils.plot_spectrogram(spectrogram, None,
                                color_map_name='parula',
                                color_map_path='/mafat/sota-mafat-radar/data/cmap.npy',
                                save_path=os.path.join(output_path,
                                    '{}_{}.png'.format(sid, data['target_type'][sid])))


if '__main__' == __name__:
    visualize_a_few(**vars(get_args()))
