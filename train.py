from datetime import datetime
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




class NoiseDataset():
    def __init__(self, imgs_dataset, noise_schedule = None):
        self.imgs_dataset = imgs_dataset
        self.noise_schedule = noise_schedule if noise_schedule else CosineNoiseAdder()       

    def __getitem__(self, idx):
        img, label = self.imgs_dataset[idx]
        t = torch.randint(self.noise_schedule.T, (1, )).squeeze()
        noisy_img, noise = self.noise_schedule.image_at_time_step(img, t)
        return noisy_img, noise, t, label

    def __len__(self):
        return len(self.imgs_dataset)


def eval_model(model:nn.Module, _test_loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (noisy_imgs, noises, time_steps, labels) in enumerate(_test_loader):
            noisy_imgs, noises, time_steps, labels = noisy_imgs.to(device), noises.to(device), time_steps.to(device), labels.to(device)
            outputs = model(noisy_imgs, time_steps, labels)
            loss = criterion(outputs, noises)
            losses.append(loss.item())
    return sum(losses)/len(losses)


def train_model(cfg: CFG, model, train, test, device=None):
    if device is None:
        # Set the device to GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    model.apply(init_weights)  # Initialize model weights

    train_loader = DataLoader(train, shuffle=True, batch_size=cfg.batch_size)
    test_loader = DataLoader(test, shuffle=False, batch_size=cfg.batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs_lr * len(train_loader), eta_min=cfg.final_lr)

    # keep track of the loss
    train_losses = []
    val_losses = []
    best_loss = np.inf
    best_epoch = 0
    stopping = False

    # Compile the model for better performance
    if cfg.use_compile:
        print("Compiling the model...")
        torch.compile(model, mode='default', dynamic=True)

    # training loop
    for epoch in range(cfg.n_epochs):
        if stopping:  # if the early stopping is triggered, we stop the training
            break
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.n_epochs}', leave=True)
        epoch_train_loss = 0

        for i, batch in enumerate(train_loader_tqdm):
            noisy_imgs, noises, time_steps, labels = batch
            noisy_imgs, noises, time_steps, labels = noisy_imgs.to(device, dtype=cfg.images_precision), noises.to(device, dtype=cfg.images_precision), time_steps.to(device, dtype=torch.int16), labels.to(device, dtype=torch.int32)
            optimizer.zero_grad()

            predicted_noise = model(noisy_imgs, time_steps, labels, verbose=False)
            loss = criterion(predicted_noise, noises)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            train_losses.append(loss.item())

        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)

        # Evaluate the model at the end of the epoch
        val_loss = eval_model(model, test_loader, criterion, device)
        val_losses.append(val_loss)

        # Print training and validation loss
        print(f'[{datetime.now().strftime(r'%d/%m, %H:%M:%S')}] - Epoch [{epoch+1}/{cfg.n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, lr: {scheduler.get_last_lr()[0]}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), cfg.save_path)

        # Early stopping
        if epoch - best_epoch > cfg.patience:
            print("Stopping early")
            stopping = True
            break

        if epoch < cfg.n_epochs_lr:
            scheduler.step()

    # Plot the losses by averaging the train loss for each epoch
    plt.figure(figsize=(10, 5))
    n = len(val_losses)
    b = int(len(train_losses)/n)
    average_train_losses =  np.array(train_losses)
    average_train_losses = [average_train_losses[i:i+b].mean() for i in range(n)]
    plt.plot(range(1, n + 1), average_train_losses, label='Training Loss')
    plt.plot(range(1, n + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    cfg_path = 'configs/config_CELEBA.yaml'

    cfg = CFG()
    cfg.from_yaml(cfg_path)
    print(cfg)

    
    CosineNoise = CosineNoiseAdder(cfg.max_time_steps)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.Normalize((0.5, 0.5, 0.5, ), (0.5, 0.5, 0.5, ))
    ])

    train = NoiseDataset(Dataset(cfg.dataset_name, transform=trans), CosineNoise)
    test = NoiseDataset(Dataset(cfg.dataset_name, 'test', transform=trans), CosineNoise)

    model = Unet(
        first_hidden=cfg.first_hidden, depth=cfg.depth, embed_dim=cfg.embedding_dim, 
        num_label=cfg.num_labels, initial_channels=cfg.initial_channels, 
        conv_layers=cfg.conv_layers, dropout=cfg.dropout
    )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using defice : ', device)
    train_model(cfg, model, train, test, device=device)

