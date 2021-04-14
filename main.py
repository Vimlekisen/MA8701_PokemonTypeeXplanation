import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split
from PIL import Image
import shap
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time
import copy

# Custom imports
from dataloaders import *
from plotutils import *
from tests import *
import config as cfg

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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    '''
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    '''
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


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
    img_filenames = np.array(os.listdir(img_dir))
    df_pokemon['Image'] = img_filenames

    # Differentiate between .png and .jpg as they need different processing when loading.
    df_pokemon['Valid'] = df_pokemon['Image'].apply(lambda x: 0 if x[-2]=='p' else 1)

    # Shuffle those indexes
    np.random.seed(42)
    shuffle_idx = np.arange(len(df_pokemon))
    np.random.shuffle(shuffle_idx)

    df_pokemon.reindex(shuffle_idx)
    img_filenames = img_filenames[shuffle_idx]


    # Define train/val/test split points
    split = int(0.8*len(df_pokemon))
    valtest_split = split + int(0.1*len(df_pokemon))


    df_train = df_pokemon.iloc[:split, :].reset_index(drop=True)
    df_val = df_pokemon.iloc[split:valtest_split, :].reset_index(drop=True)
    df_test = df_pokemon.iloc[valtest_split:, :].reset_index(drop=True)

    batch_size = 32
    num_workers = 1
    shuffle = True
    pin_memory = True
    targets = ['Type1', 'Type2']

    train_dataset = DatasetPokemon(df_train, img_dir, targets=targets, is_train=True)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                               pin_memory=pin_memory)

    val_dataset = DatasetPokemon(df_val, img_dir, targets=targets, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_dataset = DatasetPokemon(df_test, img_dir, targets=targets, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=pin_memory)

    #dataloader_test(train_loader)
    #dataloader_test(val_loader)
    #dataloader_test(test_loader)

    #balanceDataset(train_dataset)

    ####### CREATE NEURAL NET MODEL ######
    model = models.vgg19(pretrained=cfg.pretrained)
    model.classifier[-1] = nn.Linear(4096,NUM_CLASSES)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    ###### TRAIN NEURAL NET ######
    dataloaders = {
        'train': train_loader,
        'val'  : val_loader,
        'test' : test_loader
    }

    criterion = cfg.criterion
    optimizer = cfg.optimizer(model.parameters(), weight_decay=0.1)

    train_model(model, dataloaders, criterion, optimizer, num_epochs=250)