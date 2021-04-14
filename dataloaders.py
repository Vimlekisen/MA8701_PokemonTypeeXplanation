import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from CONST import *

'''
def balanceDataset(dataset):
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

    _original =

    for i, _class in enumerate(MAP_LABELS.keys()):
        while samples_per_class[_class] < wanted:
            # with sample[idx] as seed:
            # add k samples with this augments
            # add another augment
            # ...
            # all augment types added
            # all augment types added
            samples_per_class[_class] += num_augments

            # do stuff

    print(samples_per_class)

    wanted_num_samples = np.max(samples_per_class) * 2
    # for class in dataset:
        # while class has less than N samples:
            # generate more data with augmentations

    return df
'''

class DatasetPokemon(Dataset):
    '''
    Constructs a dataset containing:
    X: images of pokemons
    y: label(s) ['Type1'] or ['Type2'] or ['Type1', 'Type2'], determined by "targets" parameter.
    Currently does not support returning the name of the pokemon as a label.
    '''
    # Credit: https://www.kaggle.com/sanketgandhi/pokemon
    def __init__(self, df, image_dir, targets=['Type1'], is_train=True):
        for t in targets:
            assert t in ['Type1', 'Type2']
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
                         T.RandomErasing(p=0.5, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=255, inplace=False),
                         T.RandomRotation(20, expand = False, center = None, fill = 255) #,
                         #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
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



        # Now to fetch the corresponding label(s)/target(s)
        label_strings = self.df.loc[index, self.targets].tolist()

        # If only one type, set the numerical NaN to a string NaN so it is accessible in MAP_LABELS.
        if label_strings[-1] not in MAP_LABELS.keys():
            label_strings[-1] = 'NaN'

        # Map labels from string values to numerical values, compatible with softmax / cross-entropy.
        labels = np.array([MAP_LABELS[l] for l in label_strings])
        label = np.int64(labels[0])

        return image, label
