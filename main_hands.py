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

def test_model(model, dataloaders, criterion):
    since = time.time()
    model.eval()

    train_acc_history = 0
    val_acc_history = 0
    test_acc_history = 0

    best_acc = 0.0

    for epoch in range(1):
        print('Testing network predictions')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'train':
                train_acc_history = epoch_acc
            if phase == 'val':
                val_acc_history = epoch_acc
            if phase == 'test':
                test_acc_history = epoch_acc

        print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Training accuracy:", train_acc_history)
    print("Validation accuracy:", val_acc_history)
    print("Test accuracy:", test_acc_history)

    # load best model weights
    model.train()
    return

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, filename=None):
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
                if filename:
                    torch.save(model.state_dict(), filename)
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
    np.random.seed(42)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on", torch.cuda.get_device_name(0), "--- device:", device)
    img_dir = './hands_data/train/'
    img_dir_test = './hands_data/test/'

    ### CREATE DATASET ###

    img_filenames = {"Image": np.array(os.listdir(img_dir))}
    img_filenames_test = np.array(os.listdir(img_dir_test))
    np.random.shuffle(img_filenames_test)
    df = pd.DataFrame(img_filenames)


    # Shuffle those indexes
    shuffle_idx = np.arange(len(df))
    np.random.shuffle(shuffle_idx)
    df.reindex(shuffle_idx)


    # Define train/val/test split points
    split = int(0.8*len(df))
    valtest_split = split + int(0.1*len(df))


    df_train = df.iloc[:split, :].reset_index(drop=True)
    df_val = df.iloc[split:, :].reset_index(drop=True)
    df_test = pd.DataFrame({"Image": img_filenames_test})

    batch_size = 32
    num_epochs = 25
    num_workers = 1
    shuffle = True
    pin_memory = True
    targets = ['num_fingers', 'left_right']
    num_classes = len(targets)*6

    train_dataset = DatasetFingers(df_train, img_dir, targets=targets, is_train=True)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                               pin_memory=pin_memory)

    val_dataset = DatasetFingers(df_val, img_dir, targets=targets, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_memory)

    test_dataset = DatasetFingers(df_test, img_dir_test, targets=targets, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)

    #dataloader_test(train_loader)
    #dataloader_test(val_loader)
    #dataloader_test(test_loader)

    #balanceDataset(train_dataset)

    ####### CREATE NEURAL NET MODEL ######
    model = models.vgg19(pretrained=cfg.pretrained)
    model.classifier[-1] = nn.Linear(4096, num_classes)

    for param in model.features.parameters():
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

    ###### SAVE / LOAD MODEL #####
    filename = "prediction_network_epochs" + str(num_epochs) + "_bs" + str(batch_size) + "_num_classes" + str(num_classes)
    load_model = os.path.exists(filename)
    #load_model = False
    if load_model:
        model.load_state_dict(torch.load(filename))
    else:
        model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, filename=filename)
        #torch.save(model.state_dict(), filename)
        print("Trained new model, rerun the script for XAI.")
        exit()


    ###### CLASSIFICATION ACCURACIES ######
    #test_model(model, dataloaders, criterion)


    ###### SHAPLEY EXPLANATIONS ######
    # select a set of background examples to take an expectation over
    batch = next(iter(test_loader))
    images, _ = batch

    background = images[5:]
    test_images = images[:5]

    # Model prediction:
    pred = model(test_images)
    print("Test predictions:", pred)

    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    # Format shap values, test images and labels for image presentation
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    labels = np.array(['0L', '1L', '2L', '3L', '4L', '5L', '0R', '1R', '2R', '3R', '4R', '5R'])
    labels = np.array([labels]*len(test_images))

    # Plot it
    shap.image_plot(shap_numpy, test_numpy, labels=labels, hspace='auto', labelpad=2)