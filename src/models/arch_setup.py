from itertools import chain
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt
from src.visualization import metrics
from src.features.augmentations import resplit_track_fixed, resplit_burst_fixed, vertical_flip, horizontal_flip
from src.features import specto_feat
from src.utils import helpers
from tqdm import *
import logging
from collections import Counter

matplotlib.use('Agg')

logger = logging.getLogger()

if helpers.isnotebook():
    mtqdm = tqdm_notebook
else:
    mtqdm = tqdm


def filter_usable_segments(data: dict) -> dict:
    """This algorithm works on the assumption that we have a Boolean Array in data['usable'] indicating whether a
    given segment is to be used for augmentation -- eligibility is set in get_track_level_data.
    It will create a list of the longest adjacent sub-tracks contained in a track and return the broken-up track as a
    list of lists, along with the corresponding doppler_burst and label arrays as lists of lists in a dictionary.

    Arguments:
            data -- {dict} -- data for one track with parameters: {'iq_sweep_burst', 'doppler_burst',
                                                                    'target_type', 'usable'}
    """
    track = data['iq_sweep_burst']
    burst = data['doppler_burst']
    # shift_segment = config.get('shift_segment', 1)
    previous_i = 0
    previous_usable = False
    tracks = []
    bursts = []
    labels = []
    for i, use in enumerate(data['usable']):
        if not use:
            if previous_usable:
                if i - previous_i > 0:
                    start = previous_i * 32
                    end = i * 32
                    tracks.append(track[:, start:end])
                    bursts.append(burst[start:end])
                    labels.append(data['target_type'][i])
                    # previous_usable = False
                    # previous_i = i+1
            previous_i = i + 1
            previous_usable = False
        else:
            previous_usable = True
    if not tracks:
        tracks.append(track)
        bursts.append(burst)
        labels.append(data['target_type'])
    return {'tracks': tracks, 'bursts': bursts, 'labels': labels}


def create_new_segments_from_splits(data_dict: dict, shift_segment: int) -> dict:
    """Splits a list of tracks into new segments of size (128, 32) by shifting the existing track in shift_segment increments
        Returns dictionary with list of segments and corresponding bursts and labels
        Arguments:
            data -- {dict} -- contains keys:  {tracks', 'bursts', 'labels'}
            shift_segment -- {int} -- the number of index steps to move for each segment split
    """
    new_segments = []
    new_bursts = []
    new_labels = []
    for i, track in enumerate(data_dict['tracks']):
        if track.shape[1] > 32:
            track = resplit_track_fixed(track=track, shift_segment=shift_segment)
            burst = resplit_burst_fixed(burst=data_dict['bursts'][i], shift_segment=shift_segment)
            new_segments.extend(track)
            new_bursts.extend(burst)
            new_labels.extend([data_dict['labels'][i]] * len(burst))
        else:
            new_segments.extend(track)
            new_bursts.extend(data_dict['bursts'][i])
            new_labels.append(data_dict['labels'][i])
    return {'segments': new_segments, 'bursts': new_bursts, 'labels': new_labels}


def create_flipped_segments(data_dict: dict, flip_type: str = 'vertical'):
    """Returns a dictionary of vertically or horizontally flipped segments
        Returns dictionary with list of flipped segments and correspondingly flipped bursts and labels
        Arguments:
            data -- {dict} -- contains keys:  {tracks', 'bursts', 'labels'}
            flip_type {str} -- indicate whether to perform horizontal or vertical flips
    """
    flip = None
    if flip_type == 'vertical':
        flip = vertical_flip
    if flip_type == 'horizontal':
        flip = horizontal_flip
    flipped_segments = []
    flipped_bursts = []
    flipped_labels = []
    for i, segment in enumerate(data_dict['segments']):
        new_segments = [flip(seg) for seg in segment]
        flipped_segments.extend(new_segments)
        new_bursts = [1 - burst for burst in data_dict['bursts'][i]]
        flipped_bursts.extend(new_bursts)
        flipped_labels.append(data_dict['labels'][i])
    return {'segments': flipped_segments, 'bursts': flipped_bursts,
            'labels': flipped_labels}


def segments_generator(data: dict, config: dict) -> dict:
    """
    Generates new and/or augmented segments according to configuration parameters.
    Returns a dictionary containing the merged set of segments
        Arguments:
            data -- {dict} -- data for one track with parameters: ['segment_id', 'geolocation_type', 'geolocation_id',
                                                                   'sensor_id', 'snr_type',
                                                                   'date_index', 'iq_sweep_burst', 'doppler_burst',
                                                                   'target_type', 'usable']
            config -- {dict}:
                 get_shifts -- {bool} -- Flag to add shifts
                 shift_segment -- {int} -- How much to shift tracks to generate new segments
                 get_horizontal_flip -- {bool} -- Flag to add horizontal flips
                 get_vertical_flip -- {bool} -- Flag to add vertical flips
                 block_size -- {int} -- Max number of samples allowed to be held in a memory
    """
    filtered_data = filter_usable_segments(data=data)
    resplit_data = create_new_segments_from_splits(filtered_data, shift_segment=config.get('shift_segment', 32))
    flips = {}
    if config.get('get_vertical_flip'):
        flips = create_flipped_segments(resplit_data, flip_type='vertical')
        resplit_data = dict(Counter(resplit_data) + Counter(flips))
    if config.get('get_horizontal_flip'):
        flips = create_flipped_segments(resplit_data, flip_type='horizontal')
        resplit_data = dict(Counter(resplit_data) + Counter(flips))
    return resplit_data


class TrackDS(Dataset):
    """Dataset for a batch of segments from one track."""

    def __init__(self, data_dict):
        """
        Arguments:
            data -- {dict} -- data for 
        """
        self.data = data_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DS2(IterableDataset):
    def __init__(self, data_records: dict, config: dict):
        """
             arguments:
             ...
             data_records -- {dict}: dictionary containing records of all concatenated tracks indexed by track_id
             config -- {dict}:
                 num_tracks -- {int} -- # of tracks to take from aux dataset
                 valratio -- {int} -- Ratio of train/val split
                 get_shifts -- {bool} -- Flag to add shifts
                 shift_segment -- {int} -- How much to shift tracks to generate new segments
                 get_horizontal_flip -- {bool} -- Flag to add horizontal flips
                 get_vertical_flip -- {bool} -- Flag to add vertical flips
                 block_size -- {int} -- Max number of samples allowed to be held in a memory
        """
        super().__init__()
        self.data = data_records
        self.config = config
        self.random_index = list(self.data.keys())
        random.shuffle(self.random_index)

    def process_data(self) -> list:
        block_size = self.config.get('block_size')
        loaders = []
        for track in self.random_index:
            data_dict = pd.DataFrame(segments_generator(self.data[track], self.config)).to_dict(orient='index')
            data = TrackDS(data_dict)
            loaders.append(DataLoader(data, batch_size=block_size, shuffle=True))
        return loaders

    # @classmethod
    # def split_data_into_tracks(cls, data, block_size=50, max_workers=2):
    #
    #     for n in range(max_workers, 0, -1):
    #         if block_size % n == 0:
    #             num_workers = n
    #             break
    #     split_size = block_size // num_workers
    #     return [cls(data, config={}) for _ in range(num_workers)]

    def __iter__(self):
        """This method currently returns a list of DataLoaders from self.process_data() in a chained but linear fashion"""
        # TODO paralellize the sampling
        # TODO build a batch by paralleling the sampling i.e. taking one batch from each DataLoader in the generator.
        return chain.from_iterable(self.process_data())


def pretty_log(log):
    for key, value in log.items():
        value_s = value if type(value) == "int" else "{:.4f}".format(value)
        print(f"{key} : {value_s}, ", end="")
    print("\n---------------------------\n")


def thresh(output, thresh_hold=0.5):
    return [0 if x < thresh_hold else 1 for x in output]


def accuracy_calc(outputs, labels):
    # print("acc1:",outputs, labels)
    preds = thresh(outputs)
    # print("acc2:",preds)
    return np.sum(preds == labels) / len(preds)


def train_epochs(tr_loader, val_loader, model, criterion, optimizer, num_epochs, device, train_y, val_y, log=None,
                 WANDB_enable=False, wandb=None):
    # If we want to run more epochs, want to keep the same log of the old model
    if log:
        training_log = log
    else:
        training_log = []

    for epoch in range(num_epochs):

        print("started training epoch no. {}".format(epoch + 1))

        tr_loss = 0
        tr_size = 0
        tr_y_hat = np.array([])
        tr_labels = np.array([])

        tk0 = mtqdm(tr_loader, total=int(len(tr_loader)))

        # train loop
        for step, batch in enumerate(tk0):

            if step % 100 == 0:
                logger.info(f"step {step}")

            data, labels = batch
            tr_labels = np.append(tr_labels, labels)

            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(data)
            snr = None  # added

            # added
            if isinstance(data, list):
                snr = data[1].to(device, dtype=torch.float32)
                data = data[0]

            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # added
            if snr:
                outputs = model(data, snr)
            else:
                outputs = model(data)

            labels = labels.view(-1, 1)
            outputs = outputs.view(-1, 1)

            loss = criterion(outputs, labels)
            loss.backward()

            tr_loss += loss.item()
            tr_size += data.shape[0]

            # if torch.cuda.is_available():
            #    tr_y_hat = np.append(tr_y_hat,outputs.detach().cpu().numpy())
            # else:
            #    tr_y_hat = np.append(tr_y_hat,outputs.detach().numpy())

            tr_y_hat = np.append(tr_y_hat, outputs.detach().cpu().numpy())

            # output_t = outputs.detach().cpu().numpy()
            # print(f"output_t:{output_t}")

            # logger.info(f"tr_y_hat:{list(tr_y_hat)}")

            optimizer.step()
            optimizer.zero_grad()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss = 0
        val_size = 0
        val_y_hat = np.array([])
        val_labels = np.array([])

        logger.info("start validation")

        # validation loop

        for step, batch in enumerate(val_loader):

            data, labels = batch
            val_labels = np.append(val_labels, labels)

            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            outputs = model(data)
            if isinstance(data, list):
                snr = data[1].to(device, dtype=torch.float32)
                data = data[0]
            data = data.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            if snr is not None:
                outputs = model(data, snr)
            else:
                outputs = model(data)
            labels = labels.view(-1, 1)
            outputs = outputs.view(-1, 1)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_size += data.shape[0]

            if torch.cuda.is_available():
                val_y_hat = np.append(val_y_hat, outputs.detach().cpu().numpy())
            else:
                val_y_hat = np.append(val_y_hat, outputs.detach().numpy())

        tr_fpr, tr_tpr, _ = roc_curve(tr_labels, tr_y_hat)
        val_fpr, val_tpr, _ = roc_curve(val_labels, val_y_hat)

        epoch_log = {'epoch': epoch + 1,
                     'loss': tr_loss,
                     'auc': auc(tr_fpr, tr_tpr),
                     'acc': accuracy_calc(tr_y_hat, tr_labels),
                     'val_loss': val_loss,
                     'val_auc': auc(val_fpr, val_tpr),
                     'val_acc': accuracy_calc(val_y_hat, val_labels)}

        pretty_log(epoch_log)
        logger.info(epoch_log)

        training_log.append(epoch_log)

        if WANDB_enable:
            wandb.log(epoch_log)

    return training_log


def plot_loss_train_test(logs, model):
    tr_loss = []
    val_loss = []
    for epoch_log in logs:
        tr_loss.append(epoch_log['loss'])
        val_loss.append(epoch_log['val_loss'])

    plt.figure(figsize=(12, 8))
    plt.title(model._get_name())
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(len(tr_loss)), tr_loss, label="Train");
    plt.plot(range(len(val_loss)), val_loss, label="Val");
    plt.legend()
    plt.tight_layout()
    plt.show();


def plot_ROC_local_gpu(train_loader, val_loader, model, device):
    """
    Working on a local GPU, there is limited space and therefore a need to run the ROC examples in batches.

    Outputs ROC plot as defined in utils.stats

    Arguments:
        train_loader -- {DataLoader} -- has train data stored in batches defined in notebook
        val_loader -- {DataLoader} -- has val data stored in batches defined in notebook
        model -- {nn.Module} -- pytorch model
        device -- {torch.device} -- cpu/cuda

    """
    tr_y = np.array([])
    tr_y_hat = np.array([])
    vl_y = np.array([])
    vl_y_hat = np.array([])

    for data, label in train_loader:
        tr_y_hat = np.append(tr_y_hat, np.array(thresh(model(data.to(device).type(torch.float32)).detach().cpu())))
        tr_y = np.append(tr_y, np.array(label.detach().cpu()))

    for data, label in val_loader:
        vl_y_hat = np.append(vl_y_hat, np.array(thresh(model(data.to(device).type(torch.float32)).detach().cpu())))
        vl_y = np.append(vl_y, np.array(label.detach().cpu()))

    pred = [tr_y_hat, vl_y_hat]
    actual = [tr_y, vl_y]
    metrics.stats(pred, actual)


def plot_ROC(train_x, val_x, train_y, val_y, model, device):
    """
    Outputs ROC plot as defined in utils.stats

    Arguments:
        train_x -- {np.array} -- train data
        val_x -- {np.array} --  val data
        train_y -- {np.array} -- train labels
        val_y -- {np.array} -- val labels
        model -- {nn.Module} -- pytorch model
        device -- {torch.device} -- cpu/cuda

    """
    x1 = thresh(model(torch.from_numpy(train_x).to(device).type(torch.float32)).detach().cpu())
    x2 = thresh(model(torch.from_numpy(val_x).to(device).type(torch.float32)).detach().cpu())

    pred = [x1, x2]

    actual = [train_y, val_y]
    metrics.stats(pred, actual)
