import matplotlib.pyplot as plt
from CONST import *


def dataloader_test(loader):
    '''
    Takes in a dataloader, samples from it and illustrates the image and the associated label.
    '''
    for x,y in loader:
        print("X: image data has type", type(x), "and size", x.size())
        print("y: label data har type", type(y), "and size", len(y))

        label = MAP_LABELS_INV[int(y[0].cpu().detach().numpy())]
        img = x[0].cpu().detach().permute(1, 2, 0).numpy()

        plt.imshow(img)
        plt.title(label)
        plt.show()
        break