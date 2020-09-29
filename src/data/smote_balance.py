from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import *
from src.models import simpleDS
from torch.utils.data import DataLoader
import numpy as np


def run_smote(train_set,train_y):
    counter_pre = Counter(train_y)

    train_x = []
    for i in tqdm(range(len(train_set))):
        train_x.append(train_set[i][0])

    train_x = np.stack( train_x, axis=0 )
    train_x1 = train_x.reshape(len(train_x),-1)

    oversample = SMOTE()
    X, y = oversample.fit_resample(train_x1, train_y)

    counter = Counter(y)

    X = X.reshape(-1,126,32,1)

    train_set = simpleDS.SimpleDS(X, y)

    print(f"pre smote:{counter_pre}")
    print(f"after smote:{counter}")
    print(f"X shape: {X.shape}")

    return train_set
