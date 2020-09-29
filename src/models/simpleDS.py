"""
simple dataset file for torch
"""

import pandas as pd
from torch.utils.data import Dataset

class SimpleDS(Dataset):
    def __init__(self,df,labels):
        super().__init__()
        self.df=df
        self.labels=labels

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = self.df[idx]
        label = self.labels[idx]
        return data,label