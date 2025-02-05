import torch

import matplotlib.pyplot as plt
import numpy as np

def show_image(img:torch.Tensor, title:str=None):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.show()
