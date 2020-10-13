# -*- coding: utf-8 -*-
"""
script to run 
"""
# %%
import configparser
from os import path
import os
import sys


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
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from matplotlib.colors import LinearSegmentedColormap
from termcolor import colored

from src.data import feat_data, get_data, get_data_pipeline
from src.data.iterable_dataset import Config, DataDict, StreamingDataset, MultiStreamDataLoader, iq_to_spectogram, \
    normalize
from src.models import arch_setup, alex_model, dataset_ram_reduced
from src.features import specto_feat, add_data
from src.visualization import metrics
from src.utils import helpers

import logging

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--num_tracks', type=int, default=3, help='num_tracks from auxilary')
parser.add_argument('--val_ratio', type=str, default=6,
                    help='from good tracks, how many to take to validation set (1:X)')
parser.add_argument('--shift_segment', type=int, default=6,
                    help='shifts to use. can be single value, a range 1-31, or comma separated values')
parser.add_argument('--get_shifts', type=bool, default=True, help='whether to add shifts')
parser.add_argument('--get_horizontal_flip', type=bool, default=True, help='whether to add horizontal flips')
parser.add_argument('--get_vertical_flip', type=bool, default=True, help='whether to add vertical flips')
parser.add_argument('--batch_size', type=int, default=50, help='batch_size')
parser.add_argument('--learn_rate', type=float, default=1e-4, help='learn_rate')
parser.add_argument('--wandb', type=bool, default=True, help='enable WANDB logging')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to run')
parser.add_argument('--full_data_pickle', type=str, default=None,
                    help='pickle file with pre-compiled full_data dataframe')
parser.add_argument('--pickle_save_fullpath', type=str, default=None,
                    help='if provided, save the full_data dataframe to a different location (should be absolute path)')
parser.add_argument('--output_data_type', type=str, default="spectrogram", help='scalogram/spectrogram')
parser.add_argument('--include_doppler', type=bool, default=True,
                    help='include the doppler in the iq matrix (for spectogram')

args = parser.parse_args()

batch_size = args.batch_size
lr = args.learn_rate
WANDB_enable = args.wandb
epochs = args.epochs

config = Config(file_path=PATH_DATA, **vars(args))

print(config)

# %%

# if you want to run WANDB. run './src/scripts/wandb_login.sh' in shell (need to run only once per session/host)

if WANDB_enable == True:
    import wandb
    wandb.login()
    runname = input("Enter WANDB runname:")
    notes = input("Enter run notes :")
    os.environ['WANDB_NOTEBOOK_NAME'] = os.path.splitext(os.path.basename(__file__))[0]

# %%
log_filename = "alexnet_pytorch.log"
if os.path.exists(log_filename):
    os.remove(log_filename)

logging.basicConfig(level=logging.INFO,
                    filename='alexnet_pytorch.log',
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

# %%

dataset = DataDict(config=config)
track_count = len(dataset.train_data) + len(dataset.val_data)
segment_count = dataset.data_df.shape[0]

train_dataset = StreamingDataset(dataset.train_data, config, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
val_data = StreamingDataset(dataset.val_data, config, is_val=True, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config['batch_size'])

# %%

model = alex_model.alex_mdf_model()
# model.apply(init_weights)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.to(device)

if WANDB_enable == False:
    wandb = None
else:
    wandb.init(project="sota-mafat-base", name=runname, notes=notes, config=config)
    os.environ['WANDB_NOTEBOOK_NAME'] = os.path.splitext(os.path.basename(__file__))[0]
    wandb.watch(model)

# %%

log = arch_setup.train_epochs(
    train_loader, val_loader, model, criterion, optimizer,
    num_epochs=epochs, device=device,
    WANDB_enable=WANDB_enable, wandb=wandb)

# %%

# this ugly code can be replaced in the future if we can do (causing error now):
#  > sample = train_dataset[0:2]

# sample_num = min(6000, len(train_dataset)) 
# sample_ids = np.random.choice(len(train_dataset), sample_num, replace=False)
# sample_x = []
# sample_y = []
# for i in range(sample_num):
#     sample_x.append(train_dataset[i]['output_array'])
#     sample_y.append(train_dataset[i]['target_type'])   
# sample_x = np.stack( sample_x, axis=0 )
# sample_y = np.stack( sample_y, axis=0 )

# val_x = []
# val_y = []
# for i in range(len(val_data)):
#     val_x.append(val_data[i]['output_array'])
#     val_y.append(val_data[i]['target_type'])
# val_x = np.stack( val_x, axis=0 )

# pred = [model(torch.from_numpy(sample_x).to(device, dtype=torch.float)).detach().cpu().numpy(),
#         model(torch.from_numpy(val_x).to(device, dtype=torch.float)).detach().cpu().numpy()]
# actual = [sample_y, val_y]
# plt1 = metrics.stats(pred, actual)
# if WANDB_enable:
#     wandb.log({"roc_chart": plt1})


# #%%

# # SUBMIT

test_path = 'MAFAT RADAR Challenge - FULL Public Test Set V1'
test_df = pd.DataFrame.from_dict(get_data.load_data(test_path, PATH_DATA), orient='index').transpose()
test_df['output_array'] = test_df['iq_sweep_burst'].progress_apply(iq_to_spectogram)
if config.get('include_doppler'):
    test_df['output_array'] = test_df.progress_apply(lambda row: specto_feat.max_value_on_doppler(row['output_array'], row['doppler_burst']), axis=1)
test_df['output_array'] = test_df['output_array'].progress_apply(normalize)
test_x = torch.from_numpy(np.stack(test_df['output_array'].tolist(), axis=0).astype(np.float32)).unsqueeze(1)
test_x = test_x.permute(0, 1, 3, 2)
test_x = test_x.repeat(1, 3, 1, 1)

# Creating DataFrame with the probability prediction for each segment
submission = pd.DataFrame()
submission[['segment_id', 'label']] = test_df[['segment_id', 'target_type']]
submission['prediction'] = model(test_x.to(device)).detach().cpu().numpy()
submission['label'] = test_df['target_type']

# Save submission
submission.to_csv('submission.csv', index=False)

#print performance stats
roc = roc_curve(submission['label'], submission['prediction'])
tr_fpr, tr_tpr, _ = roc_curve(submission['label'], submission['prediction'])
auc_score = auc(tr_fpr, tr_tpr)
print(f'AUC Score: {auc_score}, Accuracy score: {arch_setup.accuracy_calc(submission["prediction"], submission["label"])}')
