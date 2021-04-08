import matplotlib.pyplot as plt
from CONST import *


def dataloader_test(loader):
    '''
    Takes in a dataloader, samples from it and illustrates the image and the associated label.
    '''
    for x,y in loader:
        print("X: image data has type", type(x), "and size", x.size())
        print("y: label data har type", type(y), "and size", len(y))
        title = ""
        for i, l in enumerate(y):
            _type = MAP_LABELS_INV[int(l[0])]
            print("Type {} :".format(i), _type)
            title += _type + " "

        img = x[0].cpu().detach().permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(title)
        plt.show()
        break