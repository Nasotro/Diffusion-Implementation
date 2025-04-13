import torch

import matplotlib.pyplot as plt
import numpy as np

def show_image(img:torch.Tensor, title:str=None):
    npimg = img.cpu().detach().numpy()
    npimg = npimg / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()
