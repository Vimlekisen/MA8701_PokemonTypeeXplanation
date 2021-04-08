import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
import shap
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# Custom imports
from dataloaders import *
from plotutils import *
from tests import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img_dir = './data/images/'
    labels_path = './data/pokemon.csv'

    ### IMPORT DATA ###
    # First, get the labels (type1,type2) for each pokemon in the dataset.
    nRowsRead = None  # specify 'None' if want to read whole file
    df_pokemon = pd.read_csv(labels_path, delimiter=',', nrows=nRowsRead)
    df_pokemon.dataframeName = 'pokemon.csv'
    nRow, nCol = df_pokemon.shape

    # Quick visualization
    print(f'There are {nRow} rows and {nCol} columns')
    print(df_pokemon.head(5))

    # Show heatmap of type combinations
    #plotType1vsType2(df_pokemon)

    # Show barplot with Type1 occurrences.
    #plotType1(df_pokemon)

    ### CREATE DATASET ###
    df_pokemon = df_pokemon.sort_values(by=['Name'], ascending=True).reset_index(drop=True)
    img_filenames = os.listdir(img_dir)
    df_pokemon['Image'] = img_filenames

    # Differentiate between .png and .jpg as they need different processing when loading.
    df_pokemon['Valid'] = df_pokemon['Image'].apply(lambda x: 0 if x[-2]=='p' else 1)

    # Define train/val/test split points
    split = int(0.8*len(df_pokemon))
    valtest_split = split + int(0.1*len(df_pokemon))

    df_train = df_pokemon.iloc[:split, :].reset_index(drop=True)
    df_val = df_pokemon.iloc[split:valtest_split, :].reset_index(drop=True)
    df_test = df_pokemon.iloc[valtest_split:, :].reset_index(drop=True)

    batch_size = 8
    num_workers = 1
    shuffle = True
    pin_memory = True
    targets = ['Type1', 'Type2']

    train_dataset = DatasetPokemon(df_train, img_dir, targets=targets)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                               pin_memory=pin_memory)

    val_dataset = DatasetPokemon(df_val, img_dir, targets=targets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_dataset = DatasetPokemon(df_test, img_dir, targets=targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=pin_memory)

    dataloader_test(train_loader)
    dataloader_test(val_loader)
    dataloader_test(test_loader)