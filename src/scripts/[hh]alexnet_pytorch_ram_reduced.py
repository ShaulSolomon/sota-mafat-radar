# -*- coding: utf-8 -*-
"""
script to run 
"""
#%%

import configparser
import os.path
from os import path
import logging
import os
import sys

# adding cwd to path to avoid "No module named src.*" errors
sys.path.insert(0,os.path.join(os.getcwd()))

#%%

import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from matplotlib.colors import LinearSegmentedColormap
from termcolor import colored

from src.data import feat_data, get_data, get_data_pipeline
from src.models import arch_setup, alex_model, dataset_ram_reduced
from src.features import specto_feat,add_data

#%%
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

wandb = None
# if WANDB_enable == True:
#     if ENV=="COLAB":
#       !pip install --upgrade wandb
#     import wandb
#     !wandb login {config_parser['MAIN']["WANDB_LOGIN"]}
#     wandb.init(project="sota-mafat-base")
#     os.environ['WANDB_NOTEBOOK_NAME'] = '[SS]Alexnet_pytorch'

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

config = dict()
config['num_tracks'] = 0
config['val_ratio'] = 3
config['shift_segment'] = list(np.arange(1,31))
config['get_shifts'] = False
config['get_horizontal_flip'] = False
config['get_vertical_flip'] = False

batch_size = 32
lr = 1e-4

full_data_picklefile = PATH_DATA+'/full_data.pickle'
if path.exists(full_data_picklefile):
  print('getting full_data from pickle')
  with open(full_data_picklefile, 'rb') as handle:
      full_data = pickle.load(handle)
else:
  print('regenerating full_data')
  full_data = get_data_pipeline.pipeline_trainval_ram_reduced(PATH_DATA, config)
  ## SAVE TO PICKLE
  with open(full_data_picklefile, 'wb') as handle:
      pickle.dump(full_data, handle,protocol=pickle.HIGHEST_PROTOCOL)

print(len(full_data[full_data.is_validation==False]))
print(len(full_data[full_data.is_validation==True]))

#%%

train_set = dataset_ram_reduced.DS(full_data[full_data.is_validation==False])
val_set= dataset_ram_reduced.DS(full_data[full_data.is_validation==True])

train_y = np.array(full_data[full_data.is_validation==False]['target_type']=='human').astype(int)
val_y = np.array(full_data[full_data.is_validation==True]['target_type']=='human').astype(int)

train_loader=DataLoader(dataset= train_set, batch_size = batch_size, shuffle = True, num_workers = 2)
val_loader=DataLoader(dataset= val_set, batch_size = batch_size, shuffle = True, num_workers = 2)

#%%

model= alex_model.alex_mdf_model()
# model.apply(init_weights)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.to(device)

if WANDB_enable == False:
  wandb = None
else:
    runname = input("Enter WANDB runname(ENTER to skip wandb) :")
    notes = input("Enter run notes :")

    wandb.init(project="sota-mafat-base",name=runname, notes=notes)
    os.environ['WANDB_NOTEBOOK_NAME'] = '[SS]Alexnet_pytorch'
    
    wandb.watch(model)
    wandb.config['data_config'] = config
    wandb.config['train_size'] = len(full_data[full_data.is_validation==False])
    wandb.config['val_size'] = len(full_data[full_data.is_validation==True])
    wandb.config['batch_size'] = batch_size
    wandb.config['learning rate'] = lr
    wandb.log(config)

log = arch_setup.train_epochs(train_loader,val_loader,model,criterion,optimizer,num_epochs= 10,device=device,train_y=train_y,val_y=val_y, WANDB_enable = WANDB_enable, wandb= wandb)


#%%

# this ugly code can be replaced in the future if we can do (causing error now):
#  > sample = train_set[0:2]
 
sample_num = min(6000, len(train_set)) 
sample_ids = np.random.choice(len(train_set), sample_num, replace=False)
sample_x = []
sample_y = []
for i in range(sample_num):
    sample_x.append(train_set[i][0])
    sample_y.append(train_set[i][1])   
sample_x = np.stack( sample_x, axis=0 )
sample_y = np.stack( sample_y, axis=0 )

val_x = []
for i in range(len(val_set)):
    val_x.append(val_set[i][0])
val_x = np.stack( val_x, axis=0 )

pred = [model(torch.from_numpy(sample_x).to(device, dtype=torch.float)).detach().cpu().numpy(),
        model(torch.from_numpy(val_x).to(device, dtype=torch.float)).detach().cpu().numpy()]
actual = [sample_y, val_y]
metrics.stats(pred, actual)

#%%


# """## SUBMIT"""

test_path = 'MAFAT RADAR Challenge - Public Test Set V1'
test_df = get_data.load_data(test_path, PATH_DATA)
test_df = specto_feat.data_preprocess(test_df.copy())
test_x = test_df['iq_sweep_burst']
test_x = test_x.reshape(list(test_x.shape)+[1])

# Creating DataFrame with the probability prediction for each segment
submission =  pd.DataFrame()
submission['segment_id'] = test_df['segment_id']
submission['prediction'] = model(torch.from_numpy(test_x).to(device).type(torch.float32)).detach().cpu().numpy()
submission['prediction'] = submission['prediction'].astype('float')

# Save submission
submission.to_csv('submission.csv', index=False)

