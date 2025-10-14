import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader


from models.unet import Unet, init_weights

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import show_image, CFG
from utils.noise import CosineNoiseAdder

from data.dataset import Dataset

from tqdm import tqdm


def sample(model, n_samples=1, config=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config is None:
        raise ValueError("Configuration object is required.")
    if not isinstance(config, CFG):
        raise ValueError("Invalid configuration object. Expected instance of CFG.")
    
    max_time_steps = config.max_time_steps
    CosineNoise = CosineNoiseAdder(max_time_steps, s=0.008)
    all_samples = []
    all_vis = []

    for i in range(n_samples):
        with torch.no_grad():
            full_img = torch.tensor([], device=device)
            full_predicted_noise = torch.tensor([], device=device)
        
            xt = torch.randn((1, config.initial_channels, config.image_size, config.image_size)).to(device)
            for t in torch.arange(max_time_steps-1, -1, -1):
                t = t.expand((1)).to(device)
                a_t = CosineNoise.get_alpha_t(t)
                alpha_t_barre = CosineNoise.get_alpha_t_barre(t)
                sigma = torch.sqrt(1-a_t).view(1, 1, 1, 1)
                noise = torch.randn_like(xt)
                # print(xt.shape)
                label = torch.tensor([i%10], device=device)
                epsilon = model(xt, t, label)
                
                a = ((1 - a_t)/(torch.sqrt(1 - alpha_t_barre))).view(1, 1, 1, 1)
                b = (1/torch.sqrt(a_t)).view(1, 1, 1, 1)
                
                if t.item() % (max_time_steps / 10) == 0 or t.item() == max_time_steps-1:
                    # print(t.item())
                    full_img = torch.cat((full_img, xt), 3)
                    full_predicted_noise = torch.cat((full_predicted_noise, epsilon), 3)
                    # print(xt.shape)
                    # show_image(xt[0], f'{t.item()}%')
                    # show_image(full_img[0])
                
                xt = b*(xt - a*epsilon) + sigma*noise

                # xt = torch.sqrt(1 - a_t).view(1, 1, 1, 1) * noise_predicted + sigma * noise
                
                # xt = b * (xt - torch.sqrt(1-alpha_t_barre)*noise_predicted) + sigma*noise

        
            all_samples.append(xt[0].cpu())
            all_vis.append(full_img[0].cpu())
        
            # show_image(xt[0], title=f'Final Image of {label.item()}')
            # show_image(full_img[0])
            
    return all_samples, all_vis


if __name__ == "__main__":
    cfg_path = 'configs/config_CELEBA.yaml'
    config_CELEBA = CFG()
    config_CELEBA.from_yaml(cfg_path)
    print(config_CELEBA)
    
    model = Unet(
        first_hidden=config_CELEBA.first_hidden, depth=config_CELEBA.depth, embed_dim=config_CELEBA.embedding_dim, 
        num_label=config_CELEBA.num_labels, initial_channels=config_CELEBA.initial_channels, 
        conv_layers=config_CELEBA.conv_layers, dropout=config_CELEBA.dropout
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(config_CELEBA.save_path, map_location='cpu'))
    model.eval()
    model.to(device)
    
    
    n_samples = 3
    
    imgs, full_imgs = sample(model, n_samples, config_CELEBA, device=device)
    for i, (xt, full_img) in enumerate(zip(imgs, full_imgs)):
        xt = xt / 2 + 0.5  # unnormalize
        full_img = full_img / 2 + 0.5  # unnormalize
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(xt.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.title(f"Final Image of {i}")
        plt.subplot(1, 2, 2)
        plt.imshow(full_img.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.title(f"Sampling process of {i}")
        plt.show()