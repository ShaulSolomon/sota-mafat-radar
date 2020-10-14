from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from src.visualization import metrics
from src.features import specto_feat, augmentations
from tqdm import tqdm_notebook as tqdm

import logging
logger = logging.getLogger()


class DS(Dataset):
    def __init__(self,df, data_type = 'spectrogram'):
        """
        Arguments:
        df -- {dataframe} -- data. expected columns: target_type (labels), doppler_burst, iq_sweep_burst, augmentation_info
        data_type -- {str} -- indidcator for which type of data we are using as input, either spectrogram (default) or scalogram

        index is expected to be in ascending order, but it might contain holes!
        index should be the same as segment_id.
        this is important when locating the segment of the augmentations.
        """

        super().__init__()
        self.df=df
        self.data_type = data_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        data_inner = self.df.iloc[idx].copy()  # use iloc here because must get absolute row position

        if data_inner.iq_sweep_burst is None:

            iq_matrix = None
            doppler_vector = None

            for augment_info in data_inner.augmentation_info:

                # print(f"augment_info:{augment_info}")

                if augment_info['type'] == 'shift':

                    # print(f"shift")

                    iq_list = []
                    doppler_list = []
                    from_segments = augment_info['from_segments']
                    shift_by = augment_info['shift']

                    for i in from_segments:
                        iq_list.append(self.df.loc[i]['iq_sweep_burst'])     # use loc here because we need the actual segment id (by index)
                        doppler_list.append(self.df.loc[i]['doppler_burst'])

                    # print(f"iq_list:{iq_list},dopller_list:{dopller_list}. shape:{iq_list[0].shape}. len:{len(iq_list)}")

                    iq_matrix = np.concatenate(iq_list, axis=1)  # 2*(128,32) => (128,64)
                    doppler_vector = np.concatenate(doppler_list, axis=0)  # 2*(32,1) => (64,1)

                    # cut the iq_matrix according to the shift
                    iq_matrix = iq_matrix[:,shift_by:shift_by+32]
                    doppler_vector = doppler_vector[shift_by:shift_by+32]

                if iq_matrix is None and augment_info['type'] == 'flip':

                    # print(f"flip")

                    from_segment = augment_info['from_segment']

                    iq_matrix = self.df[from_segment].iq_sweep_burst
                    doppler_vector = self.df[from_segment].doppler_vector

                # print(f"iq_matrix:{iq_matrix},doppler_vector:{doppler_vector}")

            data_inner.iq_sweep_burst = iq_matrix
            data_inner.doppler_burst = doppler_vector


        # print(f"data_inner:{data_inner}")

        # convert to structure supported by preprocess method
        data_inner_o = {k:[v] for (k,v) in data_inner.to_dict().items()}
        data_inner_o['target_type'] = np.asarray(data_inner_o['target_type'])

        # print(f"data_inner3:{data_inner_o}")

        # do preprocess

        #print(f"data:{data}")

        # augementations
        # do flips (if needed)
        
        if ('augmentation_info' in data_inner_o.keys()) & (len(data_inner_o['augmentation_info'][0])>1):
            for augment_info in data_inner_o['augmentation_info'][0]: # the [0] is because we added [] in the data_inner_o
                #print(f"augment_info:{augment_info}")
                if augment_info['type']=='flip':
                    if augment_info['mode']=='vertical':
                        data_inner_o['iq_sweep_burst'] = augmentations.vertical_flip(data_inner_o['iq_sweep_burst'])
                        data_inner_o['doppler_burst'] = np.abs(128-data_inner_o['doppler_burst'])

                    if augment_info['mode']=='horizontal':
                        data_inner_o['iq_sweep_burst'] = augmentations.horizontal_flip(data_inner_o['iq_sweep_burst'])
                        data_inner_o['doppler_burst'] = np.flip(data_inner_o['doppler_burst'])

        # do preprocess
        data = data_inner_o
        data = specto_feat.data_preprocess(data_inner_o, df_type = self.data_type)
        
        data['target_type'] = np.array(int(data['target_type'][0]),dtype='int64')
        
        label2model = data['target_type']
        if self.data_type == 'spectrogram':
            data2model = np.array(data['iq_sweep_burst'])
        else:
            data2model = np.array(data['scalogram'])
        #print(f"type0:{data['target_type']}")

        # print(f"data2model:{data2model.shape}")  # (1,132,28)
        # data2model = data2model.reshape(list(data2model.shape)+[1])
        data2model = np.expand_dims(data2model.squeeze(), axis=2)  # (132,28,1)

        return torch.from_numpy(data2model.copy()), torch.tensor(label2model.astype(np.int))
