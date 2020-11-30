import datetime
import inspect
from torch.nn import functional as F
import numpy as np
import h5py
import torch
import torch.utils.data
from typing import List, Callable, Union, Tuple
import torchvision
import math
import os
import random
import json
import pdb
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
root_path = r"C:\Users\octav\PycharmProjects\Stockpred"

class Stockpred_Dataset(torch.utils.data.Dataset):


    def __init__(self,
                companies: str,
                ):
        files = [f"{root_path}/stock_data/{f}" for f in os.listdir(f"{root_path}/stock_data")]
        df = pd.DataFrame()
        for file in files:
            print(file)
            df0 = pd.read_csv(file)
            df = df.append(df0)
        df = df.dropna()

        features_considered = ['Open','High','Low','Close']
        self.features = df[features_considered]
        self.features.index = df['Date']
        self.dataset = self.features.values
        data_mean = self.dataset.mean(axis=0)
        data_std = self.dataset.std(axis=0)
        self.dataset = (self.dataset-data_mean)/data_std



    def __len__(self):
        a, _ = self.dataset.shape
        return a

    def __getitem__(self, idx: int):

        return self.dataset[idx]
    

