import os
import argparse
import shutil
import pywt

import numpy as np

import src.features.specto_feat as specto_feat
import src.visualization.specto_vis as specto_vis
import src.data.get_data as get_data
from matplotlib.animation import PillowWriter
import matplotlib.animation
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)

def calculate_scalogram(iq_matrix, flip=True):
    slow_time_scalograms = []
    for j in range(iq_matrix.shape[1]):
        coef, freqs=pywt.cwt(specto_feat.hann(iq_matrix[:, j][:, np.newaxis]), np.arange(1,8), 'cgau1')
        coef = coef[:,:,0]
        if flip:
            coef = np.flip(coef, axis=0)
        coef=np.log(np.abs(coef))
        #coef=coef[:, 1:-1]
        coef=coef.T
        slow_time_scalograms.append(coef)

    stacked_scalogram = np.stack(slow_time_scalograms)
    stacked_scalogram = np.maximum(np.median(stacked_scalogram) - 1., stacked_scalogram)
    return np.split(stacked_scalogram, stacked_scalogram.shape[0])

def some_stats(report_name, data): 
    print('{} stats:'.format(report_name), np.min(data), np.mean(data), np.max(data))

def visualize_a_few(dataset_path, output_path, few):
    np.random.seed(34)

    color_map_path = '/home/dmk0v/projects/mafat/sota-mafat-radar/data/cmap.npy'
    data = get_data.load_data('train', dataset_path)
    half = few // 2
    human_indices = np.where(data['target_type'] == 'human')[0]
    all_indices = set(range(len(data['segment_id'])))
    animal_indices = list(all_indices - set(human_indices))

    chosen_human_sids = np.random.choice(human_indices, size=half, replace=False)
    chosen_animal_sids = np.random.choice(animal_indices, size=half, replace=False)
    chosen_sids = list(chosen_human_sids) + list(chosen_animal_sids)
    for sid in chosen_sids:
        print('Sample ID ', sid)

        iq = data['iq_sweep_burst'][sid]
        #iq = iq[:, 0][:, np.newaxis]

        print('iq_sweep_burst:', iq.shape)
        print('iq_sweep_burst:', data['iq_sweep_burst'][sid][0,0])
        print('doppler_burst:', data['doppler_burst'][sid].shape)
        print('doppler_burst:', data['doppler_burst'][sid][0])
        print('target_type:', data['target_type'][sid])
        print('snr_type: {}\n'.format(data['snr_type'][sid]))

        
        spectrogram = specto_feat.calculate_spectrogram(iq)
        print(spectrogram.shape)
        some_stats('spectrogram', spectrogram)
        temp = np.zeros((126, 224))
        temp[:, :] = np.min(spectrogram)
        temp[:, :32] = spectrogram
        spectrogram = temp 

        scalogram = calculate_scalogram(iq)
        some_stats('scalogram', scalogram)

        specto_vis.plot_spectrogram(spectrogram, None,
                                    color_map_path=color_map_path,
                                    save_path=os.path.join(output_path,
                                    '{}_{}.png'.format(data['target_type'][sid], sid)))

        #specto_vis.plot_spectrogram(scalogram, None,
        #                            color_map_path=color_map_path,
        #                            save_path=os.path.join(output_path,
        #                            '{}_{}_scal.png'.format(data['target_type'][sid], sid)))

        # specto_vis.plot_spectrogram(iq.real, None,
        #                             color_map_path=color_map_path,
        #                             save_path=os.path.join(output_path,
        #                             '{}_{}_i.png'.format(data['target_type'][sid], sid)))
        # specto_vis.plot_spectrogram(iq.imag, None,
        #                             color_map_path=color_map_path,
        #                             save_path=os.path.join(output_path,
        #                             '{}_{}_q.png'.format(data['target_type'][sid], sid)))
        f = plt.figure()
        color_map = LinearSegmentedColormap.from_list('parula', np.load(color_map_path))
        scalogram_images = []
        for stamp in scalogram:
            im = plt.imshow(stamp[0, :, :], cmap=color_map)
            scalogram_images.append([im])

        ani = matplotlib.animation.ArtistAnimation(f, scalogram_images, interval=3.2, blit=True, repeat_delay=32)
        ani.save(os.path.join(output_path,'{}_{}_scal.gif'.format(data['target_type'][sid], sid)), writer=LoopingPillowWriter(fps=3))


if '__main__' == __name__:
    visualize_a_few(**vars(get_args()))