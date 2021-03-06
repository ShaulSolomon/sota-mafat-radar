# -*- coding: utf-8 -*-
"""
script to run 
"""
# %%
import configparser
import os
import sys
from os import path

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

# adding cwd to path to avoid "No module named src.*" errors
sys.path.insert(0, os.path.join(PATH_ROOT))

# %%
import argparse
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from src.data import get_data
from src.data.iterable_dataset import Config, DataDict, StreamingDataset, iq_to_spectogram, \
    normalize
from src.models import arch_setup, tcn_model3
from src.features import specto_feat

import logging

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--num_tracks', type=int, default=4, help='num_tracks from auxilary')
parser.add_argument('--val_ratio', type=str, default=6,
                    help='from good tracks, how many to take to validation set (1:X)')
parser.add_argument('--shift_segment', type=int, default=2,
                    help='shifts to use. can be single value, a range 1-31, or comma separated values')
parser.add_argument('--get_shifts', type=bool, default=True, help='whether to add shifts')
parser.add_argument('--get_horizontal_flip', type=bool, default=True, help='whether to add horizontal flips')
parser.add_argument('--get_vertical_flip', type=bool, default=False, help='whether to add vertical flips')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--learn_rate', type=float, default=1e-3, help='learn_rate')
parser.add_argument('--wandb', type=bool, default=True, help='enable WANDB logging')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to run')
parser.add_argument('--full_data_pickle', type=str, default=None,
                    help='pickle file with pre-compiled full_data dataframe')
parser.add_argument('--pickle_save_fullpath', type=str, default=None,
                    help='if provided, save the full_data dataframe to a different location (should be absolute path)')
parser.add_argument('--output_data_type', type=str, default="spectrogram", help='scalogram/spectrogram')
parser.add_argument('--include_doppler', type=bool, default=True,
                    help='include the doppler in the iq matrix (for spectogram')
parser.add_argument('--shuffle_stream', type=bool, default=True,
                    help='Shuffle the track streaming')
parser.add_argument('--tracks_in_memory', type=int, default=100,
                    help='How many tracks to keep in memory before flushing')
parser.add_argument('--include_test_data', type=bool, default=False,
                    help='Include the complete test dataset into the train/val.')
parser.add_argument('--mother_wavelet', type=str, default="cgau1",
                    help='Mother wavelet transformation to use when creating scalograms')
parser.add_argument('--scale', type=int, default=8,
                    help='Number of scales for creating scalograms')
parser.add_argument('--num_channels', type=list, default=[16,32,32,64],
                    help='Number of Channels for TCN Blocks')
parser.add_argument('--kernel_size', type=int, default=5,
                    help='Kernel size')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate')
                    


args = parser.parse_args()

batch_size = args.batch_size
lr = args.learn_rate
WANDB_enable = args.wandb
epochs = args.epochs

config = Config(file_path=PATH_DATA, **vars(args))

print(config)

# if you want to run WANDB. run './src/scripts/wandb_login.sh' in shell (need to run only once per session/host)

if WANDB_enable == True:
    import wandb
    wandb.login()
    runname = input("Enter WANDB runname:")
    notes = input("Enter run notes :")
    os.environ['WANDB_NOTEBOOK_NAME'] = os.path.splitext(os.path.basename(__file__))[0]

log_filename = "tcn_pytorch_2D.log"
if os.path.exists(log_filename):
    os.remove(log_filename)

logging.basicConfig(level=logging.INFO,
                    filename='tcn_pytorch_2D.log',
                    format="%(asctime)s [%(levelname)s]|%(module)s:%(message)s", )
logging.info("start")
logger = logging.getLogger()

# Set seed for reproducibility of results
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu:0')

dataset = DataDict(config=config)
track_count = len(dataset.train_data) + len(dataset.val_data)
segment_count = dataset.data_df.shape[0]

train_dataset = StreamingDataset(dataset.train_data, config, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
val_data = StreamingDataset(dataset.val_data, config, is_val=True, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config['batch_size'])

model = tcn_model3.TemporalConvNet(num_inputs = 1, num_channels = args.num_channels, kernel_size=args.kernel_size, dropout=args.dropout)
# model.apply(init_weights)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

model.to(device)

if WANDB_enable == False:
    wandb = None
else:
    wandb.init(project="sota-mafat-tcn", name=runname, notes=notes, config=config)
    os.environ['WANDB_NOTEBOOK_NAME'] = os.path.splitext(os.path.basename(__file__))[0]
    wandb.watch(model)

log = arch_setup.train_epochs(
    train_loader, val_loader, model, criterion, optimizer,
    num_epochs=epochs, device=device,
    WANDB_enable=WANDB_enable, wandb=wandb)

# # SUBMIT
test_path = 'MAFAT RADAR Challenge - FULL Public Test Set V1'
test_df = pd.DataFrame.from_dict(get_data.load_data(test_path, PATH_DATA), orient='index').transpose()
test_df.replace({'animal': 0, 'human': 1}, inplace=True)
test_df['output_array'] = test_df['iq_sweep_burst'].progress_apply(iq_to_spectogram)
if config.get('include_doppler'):
    test_df['output_array'] = test_df.progress_apply(lambda row: specto_feat.max_value_on_doppler(row['output_array'], row['doppler_burst']), axis=1)
test_df['output_array'] = test_df['output_array'].progress_apply(normalize)
test_x = torch.from_numpy(np.stack(test_df['output_array'].tolist(), axis=0).astype(np.float32)).unsqueeze(1)

# Creating DataFrame with the probability prediction for each segment
submission = pd.DataFrame()
submission[['segment_id', 'label']] = test_df[['segment_id', 'target_type']]
submission['prediction'] = model(test_x.to(device)).detach().cpu().numpy()
submission['label'] = test_df['target_type']

# Save submission
submission.to_csv('2D-submission.csv', index=False)
if WANDB_enable:
    wandb.save('2D-submission.csv')

#print performance stats
roc = roc_curve(submission['label'], submission['prediction'])
tr_fpr, tr_tpr, _ = roc_curve(submission['label'], submission['prediction'])
auc_score = auc(tr_fpr, tr_tpr)
print(f'AUC Score: {auc_score}, Accuracy score: {arch_setup.accuracy_calc(submission["prediction"], submission["label"])}')
