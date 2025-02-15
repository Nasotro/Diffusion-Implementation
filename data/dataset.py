import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


class CIFAR10_Dataset():
    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform if transform else transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.load_set()
        

    def load_set(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.set = torchvision.datasets.CIFAR10(root='./data', train = (self.split=='train'),
                                        download=True, transform=transform)
     
    def show_image(self, idx):
        img = self[idx]
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    def __getitem__(self, idx:int) -> torch.Tensor:
        """
        Returns a Tensor of shape (3 x 32 x 32)
        """
        return self.set[idx]
        
    def __len__(self) -> int:
        return len(self.set)
    
    def __str__(self) -> str:
        return f"CIFAR dataset of size {len(self)}"
    def __repr__(self) -> str:
        return str(self)


class MNIST_Dataset():
    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform if transform else transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.load_set()
        

    def load_set(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        self.set = torchvision.datasets.MNIST(root='./data', train = (self.split=='train'),
                                        download=True, transform=transform)
     
    def show_image(self, idx):
        img = self[idx]
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    def __getitem__(self, idx:int) -> torch.Tensor:
        """
        Returns a Tensor of shape (1 x 28 x 28)
        """
        return self.set[idx]
        
    def __len__(self) -> int:
        return len(self.set)
    
    def __str__(self) -> str:
        return f"MNIST dataset of size {len(self)}"
    def __repr__(self) -> str:
        return str(self)


