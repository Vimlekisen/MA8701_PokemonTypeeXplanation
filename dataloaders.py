import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd

from CONST import *


def balanceDataset(df):
    # Solution BAD: chuck out randomly from classes
    # Solution Better(?): Find average number of entries in each class, and do data augmentation for the ones that have less than average
    # Solution Best: Do data augmentation for all classes but distributed proportionally to the amount of data per class.

    # List of augments:
    #T.RandomHorizontalFlip(p=0.5)
    #T.RandomRotation(degrees, interpolation=<InterpolationMode.NEAREST: 'nearest'>, expand=False, center=None, fill=0, resample=None)
    #T.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=<InterpolationMode.BILINEAR: 'bilinear'>)
    #T.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    #T.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    #T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

    # calculate the wanted number of samples of data, e.g. 2x number of samples of most common class
    #samples_per_class = np.zeros(NUM_CLASSES)
    #for index, sample in df.iterrows():
    #    #print(type(sample), sample)
    #    _type = sample["Type1"]
    #    samples_per_class[MAP_LABELS[_type]] += 1
    #print(samples_per_class)

    samples_per_class = df['Type1'].value_counts()
    wanted = 2* samples_per_class[0]

    df_new = df.copy(deep=False)

    for i, _class in enumerate(MAP_LABELS.keys()):
        if _class == "NaN": continue

        pokemon_to_add = wanted // samples_per_class[_class]
        copies = df.loc[df['Type1'] == _class]
        for j in range(pokemon_to_add-1):
            # add a copy of all _class type pokemons to the new dataset
            df_new = pd.concat([df_new, copies])


    return df_new


class DatasetPokemon(Dataset):
    '''
    Constructs a dataset containing:
    X: images of pokemons
    y: label(s) ['Type1'] or ['Type2'] or ['Type1', 'Type2'], determined by "targets" parameter.
    Currently does not support returning the name of the pokemon as a label.
    '''
    # Credit: https://www.kaggle.com/sanketgandhi/pokemon
    def __init__(self, df, image_dir, targets=['Type1'], is_train=True):
        #for t in targets:
        #    assert t in ['Type1', 'Type2']
        self.df = df
        self.image_dir = image_dir
        self.targets = targets
        self.transform = T.Compose([T.RandomApply([
                         #transforms.RandomCrop(32, padding=4),
                         #T.RandomHorizontalFlip(p=0.5)
                         #T.RandomRotation(degrees, interpolation=<InterpolationMode.NEAREST: 'nearest'>, expand=False, center=None, fill=0, resample=None)
                         # T.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=<InterpolationMode.BILINEAR: 'bilinear'>)
                         T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                         T.GaussianBlur(11, sigma=(0.1, 2.0))


                         #T.ToTensor(),
                         ], p=0.9),
                         T.RandomHorizontalFlip(),
                         #T.RandomErasing(p=0.5, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=255, inplace=False),
                         T.RandomRotation(20, expand = False, center = None, fill = 255)
                         #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be careful for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        img_path = self.image_dir + self.df['Image'][index]
        image = Image.open(img_path)

        # If image type is png, and it has a alpha channel (transparency), then we need to define that the transparent
        # background is white. Otherwise, it will appear black.
        # Adapted from: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
        if image.mode == 'RGBA':
            _image = Image.new('RGB', image.size, (255, 255, 255))
            _image.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            image = _image
        # Otherwise, it's .jpg or .png without an alpha channel, and can be converted directly to RGB.
        else:
            image = image.convert('RGB')

        # To numpy and normalize, then make it a PyTorch Tensor and transpose the image.
        image = np.asarray(image, dtype=np.float32) / 255
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        # Data augmentation
        if self.is_train:
            image = self.transform(image)

        # image form : 120x120x3
        #mean = [0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]

        image = T.Normalize(mean=[0.7232975, 0.71917904, 0.7124889], std=[0.17589469, 0.17604435, 0.18946176])(image)
        # mean = [0.7232975  0.71917904 0.7124889 ]
        # std  = [0.17589469 0.17604435 0.18946176]

        # Now to fetch the corresponding label(s)/target(s)
        label_strings = self.df.loc[index, self.targets].tolist()

        # If only one type, set the numerical NaN to a string NaN so it is accessible in MAP_LABELS.
        #if label_strings[-1] not in MAP_LABELS.keys():
        #    label_strings[-1] = 'NaN'

        # Map labels from string values to numerical values, compatible with softmax / cross-entropy.
        #labels = np.array([MAP_LABELS[l] for l in label_strings])
        #label = np.int64(labels[0])

        labels = np.array([float(l) for l in label_strings])
        return image, labels



class DatasetFingers(Dataset):
    '''
    Constructs a dataset containing:
    X: images of hands with 0-5 fingers showing
    y: label 0-5, determined by the last two characters of the filename (2L: 2 fingers, left hand).
    '''
    # Credit: https://www.kaggle.com/sanketgandhi/pokemon
    def __init__(self, df, image_dir, targets=['num_fingers'], is_train=True):
        for t in targets:
            assert t in ['num_fingers', 'left_right']

        self.df = df
        self.image_dir = image_dir
        self.targets = targets
        self.transform = None
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be careful for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        img_name = self.df['Image'][index]
        img_path = self.image_dir + img_name
        image = Image.open(img_path)

        # If image type is png, and it has a alpha channel (transparency), then we need to define that the transparent
        # background is white. Otherwise, it will appear black.
        # Adapted from: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
        if image.mode == 'RGBA':
            _image = Image.new('RGB', image.size, (255, 255, 255))
            _image.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            image = _image
            print("RGBA")
        # Otherwise, it's .jpg or .png without an alpha channel, and can be converted directly to RGB.
        else:
            image = image.convert('RGB')

        # To numpy and normalize, then make it a PyTorch Tensor and transpose the image.
        image = np.asarray(image, dtype=np.float32) / 255
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        # Data augmentation
        if self.is_train and self.transform:
            image = self.transform(image)

        # Now to fetch the corresponding label(s)/target(s)
        # Labels are embedded in the image filenames: blablablabla____2L.png <-- indicating 2 fingers on left hand.
        label_string = img_name[-6:-4]
        num_fingers = int(label_string[0])
        hand        = 6 if label_string[1] == 'R' else 0

        # Set label to 0-11 if target is #finger+which hand, 0-5 if only #fingers is wanted.
        if len(self.targets) > 1:
            label = np.int64(num_fingers + hand)
        else:
            label = np.int64(num_fingers)

        return image, label
