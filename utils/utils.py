import torch
import yaml

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass


def show_image(img:torch.Tensor, title:str=None, unnormalize:bool=True, transpose:bool=True) -> None:
    if not isinstance(img, np.ndarray):
        npimg = img.cpu().detach().numpy()
    else: 
        npimg = img
        
    if unnormalize:
        npimg = npimg / 2 + 0.5     # unnormalize
    if transpose:
        npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


@dataclass
class CFG:
    """Configuration class for model hyperparameters."""
    # Training parameters
    n_epochs: int = 30
    lr: float = 1e-3
    final_lr: float = 1e-5
    batch_size: int = 32
    n_epochs_lr: int = 20  # epochs with learning rate reduction
    patience: int = 7000  # early stopping patience
    eval_frequency: int = 0.1  # fraction of epoch between evaluations (0.1 = 10 evals per epoch)
    
    # Model parameters
    first_hidden: int = 32
    depth: int = 3
    conv_layers: int = 3
    embedding_dim: int = 128
    num_labels: int = 10
    dropout: float = 0.2
    initial_channels: int = 1  # 1 for MNIST, 3 for CIFAR10
    
    # Dataset parameters
    max_time_steps: int = 200
    dataset_name: str = "MNIST"
    image_size: int = 32
    
    # Other parameters
    images_precision = torch.float32  # dtype for images
    
    save_path: str = "best_model.pth"
    use_compile: bool = True
    
    
    def from_yaml(self, path:str) -> None:
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, type(getattr(self, key))(value))
                else:
                    raise ValueError(f"Unknown configuration key: {key}")
    
    
    def to_yaml(self, path:str) -> None:
        """Save configuration to a YAML file."""
        config = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        with open(path, 'w') as f:
            yaml.safe_dump(config, f)
