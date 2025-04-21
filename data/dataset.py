import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


class Dataset():
    def __init__(self, name, split='train', transform=None):
        datasets = {
            'CIFAR10': torchvision.datasets.CIFAR10,
            'MNIST': torchvision.datasets.MNIST,
            'CELEBA': torchvision.datasets.CelebA,
        }
        if name not in datasets:
            raise ValueError(f"Unknown dataset {name}. Known datasets are {datasets.keys()}")
        if transform is None:
            raise ValueError("Transform should be provided")
        if split not in ['train', 'test']:
            raise ValueError("Split should be either 'train' or 'test'")
        
        self.split = split
        self.name = name
        self.transform = transform
        if name in ['CIFAR10', 'MNIST']:
            self.dataset = datasets[name](root='./data', train = (self.split=='train'), 
                                      download=True, transform=self.transform)
        elif name == 'CELEBA':
            self.dataset = datasets[name](root='./data', split=self.split, 
                                      download=True, transform=self.transform)
        self.classes = self.dataset.classes if hasattr(self.dataset, 'classes') else None

    def show_image(self, idx):
        img, label = self[idx]
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(f"Label: {self.classes[label] if self.classes else label}")
        plt.show()
    
    def __getitem__(self, idx:int) -> torch.Tensor:
        """Returns a tuple of (image, label)"""
        return self.dataset[idx]
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __str__(self) -> str:
        return f"Dataset of size {len(self)}"
    def __repr__(self) -> str:
        return str(self)


class CIFAR10_Dataset():
    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform if transform else transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.load_set()
        

    def load_set(self):
        self.set = torchvision.datasets.CIFAR10(root='./data', train = (self.split=='train'),
                                        download=True, transform=self.transform)
        # self.set = torchvision.datasets.ImageNet(root='./data', split=self.split,
        #                                 download=True, transform=transform)
     
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
        self.set = torchvision.datasets.MNIST(root='./data', train = (self.split=='train'),
                                        download=True, transform=self.transform)
     
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


