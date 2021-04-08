import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from CONST import *

class DatasetPokemon(Dataset):
    '''
    Constructs a dataset containing:
    X: images of pokemons
    y: label(s) ['Type1'] or ['Type2'] or ['Type1', 'Type2'], determined by "targets" parameter.
    Currently does not support returning the name of the pokemon as a label.
    '''
    # Credit: https://www.kaggle.com/sanketgandhi/pokemon
    def __init__(self, df, image_dir, targets=['Type1']):
        for t in targets:
            assert t in ['Type1', 'Type2']
        self.df = df
        self.image_dir = image_dir
        self.targets = targets

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
            image.load()
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

        # Now to fetch the corresponding label(s)/target(s)
        label_strings = self.df.loc[index, self.targets].tolist()

        # If only one type, set the numerical NaN to a string NaN so it is accessible in MAP_LABELS.
        if label_strings[-1] not in MAP_LABELS.keys():
            label_strings[-1] = 'NaN'

        # Map labels from string values to numerical values, compatible with softmax / cross-entropy.
        labels = [MAP_LABELS[l] for l in label_strings]

        return image, labels
