# %%
# import configparser
from os import path
import os
import sys

from torch.utils.data import DataLoader
import configparser
from tqdm import tqdm

PATH_ROOT = ""
PATH_DATA = ""

creds_path_ar = ["../../credentials.ini", "credentials.ini"]

for creds_path in creds_path_ar:
    if path.exists(creds_path):
        config_parser = configparser.ConfigParser()
        config_parser.read(creds_path)
        PATH_ROOT = config_parser['MAIN']["PATH_ROOT"]
        PATH_DATA = config_parser['MAIN']["PATH_DATA"]
        WANDB_enable = config_parser['MAIN']["WANDB_ENABLE"] == 'TRUE'
        ENV = config_parser['MAIN']["ENV"]

# %%
# adding cwd to path to avoid "No module named src.*" errors
sys.path.insert(0, os.path.join(PATH_ROOT))
from src.data.iterable_dataset import Config, DataDict, StreamingDataset, MultiStreamDataLoader

config = Config(file_path=PATH_DATA, num_tracks=3, valratio=6, get_shifts=True, output_data_type='spectrogram',
                get_horizontal_flip=True, get_vertical_flip=True, mother_wavelet='cgau1', wavelet_scale=3,
                batch_size=50, tracks_in_memory=2, include_doppler=True, shift_segment=16, shuffle_stream=True)

dataset = DataDict(config=config)
track_count = len(dataset.train_data) + len(dataset.val_data)
train_dataset = StreamingDataset(dataset.train_data, config, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
val_data = StreamingDataset(dataset.val_data, config, is_val=True, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config['batch_size'])
segment_count = len(train_dataset) + len(val_data)
segment_ids = []
train_sample_counts = []
for sample in tqdm(train_loader):
    sample_len = len(sample['target_type'])
    train_sample_counts.append(sample_len)
    segment_ids.append(sample['segment_id'])
val_sample_counts = [len(sample['target_type']) for sample in val_loader]
sample_counts = val_sample_counts + train_sample_counts
count = sum(sample_counts)
print(f'Total segments generated: {count}')
print(f'Total segments expected: {segment_count}')
print(f'Segments missing: {segment_count - count}')
print(f'Tracks created: {track_count}')
