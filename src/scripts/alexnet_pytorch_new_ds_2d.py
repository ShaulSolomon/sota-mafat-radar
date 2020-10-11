# -*- coding: utf-8 -*-
"""
script to run 
"""
#%%
import configparser
from os import path
import os
import sys

PATH_ROOT = ""
PATH_DATA = ""

creds_path_ar = ["../../credentials.ini","credentials.ini"]
PATH_ROOT = ""
PATH_DATA = ""

for creds_path in creds_path_ar:
    if path.exists(creds_path):
        config_parser = configparser.ConfigParser()
        config_parser.read(creds_path)
        PATH_ROOT = config_parser['MAIN']["PATH_ROOT"]
        PATH_DATA = config_parser['MAIN']["PATH_DATA"]
        WANDB_enable = config_parser['MAIN']["WANDB_ENABLE"] == 'TRUE'
        ENV = config_parser['MAIN']["ENV"]

# adding cwd to path to avoid "No module named src.*" errors
sys.path.insert(0,os.path.join(PATH_ROOT))

#%%
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
from src.data.iterable_dataset import Config, DataDict, StreamingDataset, MultiStreamDataLoader
from src.models import arch_setup, alex_model, dataset_ram_reduced
from src.features import specto_feat,add_data
from src.visualization import metrics
from src.utils import helpers

import logging


#%%

parser = argparse.ArgumentParser()
parser.add_argument('--num_tracks', type=int, default=3,  help='num_tracks from auxilary')
parser.add_argument('--val_ratio',  type=str, default=3, help='from good tracks, how many to take to validation set (1:X)')
parser.add_argument('--shift_segment', type=str, help='shifts to use. can be single value, a range 1-31, or comma separated values')
parser.add_argument('--get_shifts', type=bool, default=False, help='whether to add shifts')
parser.add_argument('--get_horizontal_flip', type=bool, default=False, help='whether to add horizontal flips')
parser.add_argument('--get_vertical_flip', type=bool, default=False, help='whether to add vertical flips')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--learn_rate', type=float, default=1e-4, help='learn_rate')
parser.add_argument('--wandb', type=bool, default=False, help='enable WANDB logging')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to run')
parser.add_argument('--full_data_pickle', type=str, default=None, help='pickle file with pre-compiled full_data dataframe')
parser.add_argument('--pickle_save_fullpath', type=str, default=None, help='if provided, save the full_data dataframe to a different location (should be absolute path)')
parser.add_argument('--output_data_type', type=str, default="spectogram", help='scalogram/spectogram')
parser.add_argument('--include_doppler', type=bool, default=True, help='include the doppler in the iq matrix (for spectogram')

args = parser.parse_args()

#%%

epochs = 10
batch_size = 32
lr = 1e-4
full_data_pickle = 'full_data.pickle'
pickle_save_fullpath = None

config = Config(
    file_path=PATH_DATA, num_tracks=3, valratio=6, get_shifts=True, 
    output_data_type='spectogram',get_horizontal_flip=True, get_vertical_flip=True, 
    mother_wavelet='cgau1', wavelet_scale=3, batch_size=50)

if 'args' in globals():
    batch_size = args.batch_size
    lr = args.learn_rate
    WANDB_enable = args.wandb
    epochs = args.epochs
    full_data_pickle = args.full_data_pickle
    if full_data_pickle is not None and not path.exists(f"{PATH_DATA}/{full_data_pickle}"):
        print("args pickle file doesn't exists. abort...")
        sys.exit()
    
    config['num_tracks'] = args.num_tracks
    config['val_ratio'] = args.val_ratio   
    config['get_shifts'] = args.get_shifts
    config['get_horizontal_flip'] = args.get_horizontal_flip
    config['get_vertical_flip'] = args.get_vertical_flip
    config['output_data_type'] = args.output_data_type
    config['include_doppler'] = args.include_doppler
    if args.shift_segment is not None:
        config['shift_segment'] = helpers.parse_range_list(args.shift_segment)
    if args.pickle_save_fullpath is not None:
        pickle_save_fullpath = f"{args.pickle_save_fullpath}/{full_data_pickle}"  

print(config)

#%%

# if you want to run WANDB. run './src/scripts/wandb_login.sh' in shell (need to run only once per session/host)

if WANDB_enable == True:
    print("wandb install and login start")    
    #subprocess.check_output(['sudo','./src/scripts/wandb_login.sh'])
    import wandb
    runname = input("Enter WANDB runname:")
    notes = input("Enter run notes :")
    os.environ['WANDB_NOTEBOOK_NAME'] = os.path.splitext(os.path.basename(__file__))[0]	


#%%
log_filename = "alexnet_pytorch.log"
if os.path.exists(log_filename):
    os.remove(log_filename)

logging.basicConfig(level=logging.INFO,
                    filename='alexnet_pytorch.log',
                    format="%(asctime)s [%(levelname)s]|%(module)s:%(message)s",)
logging.info("start")
logger = logging.getLogger()
 
# Set seed for reproducibility of results
seed_value = 0
os.environ['PYTHONHASHSEED']=str(seed_value)

random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu:0')

#%%

dataset = DataDict(config=config)
track_count = len(dataset.train_data) + len(dataset.val_data)
segment_count = dataset.data_df.shape[0]

train_dataset = StreamingDataset(dataset.train_data, config, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
val_data = StreamingDataset(dataset.val_data, config, is_val=True, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config['batch_size'])


#%%

model= alex_model.alex_mdf_model()
# model.apply(init_weights)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.to(device)

if WANDB_enable == False:
  wandb = None
else:
    wandb.init(project="sota-mafat-base",name=runname, notes=notes)
    os.environ['WANDB_NOTEBOOK_NAME'] = '[SS]Alexnet_pytorch'
    
    wandb.watch(model)
    wandb.config['data_config'] = config
    wandb.config['train_size'] = len(full_data[full_data.is_validation==False])
    wandb.config['val_size'] = len(full_data[full_data.is_validation==True])
    wandb.config['batch_size'] = batch_size
    wandb.config['learning rate'] = lr
    wandb.log(config)

#%%

log = arch_setup.train_epochs(
    train_loader,val_loader,model,criterion,optimizer,
    num_epochs= epochs,device=device, 
    WANDB_enable = WANDB_enable, wandb= wandb)


#%%

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

# test_path = 'MAFAT RADAR Challenge - Public Test Set V1'
# test_df = get_data.load_data(test_path, PATH_DATA)
# test_df = specto_feat.data_preprocess(test_df.copy())
# test_x = test_df['iq_sweep_burst']
# test_x = test_x.reshape(list(test_x.shape)+[1])

# # Creating DataFrame with the probability prediction for each segment
# submission =  pd.DataFrame()
# submission['segment_id'] = test_df['segment_id']
# submission['prediction'] = model(torch.from_numpy(test_x).to(device).type(torch.float32)).detach().cpu().numpy()
# submission['prediction'] = submission['prediction'].astype('float')

# # Save submission
# submission.to_csv('submission.csv', index=False)
