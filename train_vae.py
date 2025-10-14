from models.VAE import VAE
from data.dataset import MNIST_Dataset, CIFAR10_Dataset, Dataset
from utils.utils import show_image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb

def train_vae(model, train_loader,optimizer, epochs=20):
    vae_losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, _ in progress_bar:
            # print(images.shape)
            images = images.to(device)
            recon, mu, logvar, z = model(images)

            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta=cfg['beta'], epoch=epoch, max_anneal_epochs=cfg['n_epochs_kl_annealing'])
            # show_image(images[0], title=f"Input Image, loss : {loss.item():.4f}")
            # show_image(recon[0], title=f"Reconstructed Image, loss : {loss.item():.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), recon_loss=recon_loss.item(), kl_loss=kl_loss.item())
            
            latent_mean = torch.mean(z, dim=[0, 2, 3]).mean().item()  # Should be ~0
            latent_std = torch.std(z, dim=[0, 2, 3]).mean().item()  # Should be ~1
            latent_l2 = torch.norm(z, p=2, dim=1).mean().item()  # Should be ~√(latent_dim)
            wandb.log({
                "batch/total_loss": loss.item(),
                "batch/recon_loss": recon_loss.item(),
                "batch/kl_loss": kl_loss.item(),
                "batch/lr": scheduler.get_last_lr()[0],
                "latent_space/mean":latent_mean,
                "latent_space/std":latent_std,
                "latent_space/norm_l2":latent_l2,
            })

            losses['total'].append(loss.item())
            losses['recon'].append(recon_loss.item())
            losses['kl'].append(kl_loss.item())

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        vae_losses.append(avg_epoch_loss)

        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch/total_loss": avg_epoch_loss,
            "epoch/recon_loss": recon_loss.item(),
            "epoch/kl_loss": kl_loss.item(),
            "epoch/lr": scheduler.get_last_lr()[0]
        })
        
        if epoch % 2 == 0:
            wandb.log({
                "input_image": wandb.Image(images[0].cpu()),
                "reconstructed_image": wandb.Image(recon[0].cpu())
            })
            
            # break
        vae_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {vae_losses[-1]:.4f}, recon_loss: {recon_loss.item():.4f}, kl_loss: {kl_loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()
    return losses


if __name__ == '__main__':
    cfg = {
        "image_size": 64,
        "channels": 3,
        "latent_dim": 64,
        "dataset": "celeba",
        "batch_size": 32,
        "num_epochs": 300,
        "learning_rate": 1e-3,
        "n_epochs_kl_annealing": 30,
        "depth": 3, # latent img size : 8 x 8
        "beta": 1.0
    }

    model_path = f"vae_{cfg['dataset']}.pth"

    run = wandb.init(
        entity="lorrain",
        project="Diffusion VAE",
        config=cfg
    )

    latent_dim = cfg['latent_dim']
    c = cfg['channels']
    depth = cfg['depth']

    model = VAE(latent_dim=latent_dim, initial_channels=c, depth=depth)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'using device : {device}')

    img_size = cfg['image_size']
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    # train = CIFAR10_Dataset(split='train', transform=transform)
    # train = MNIST_Dataset(split='train', transform=transform)
    train = Dataset('CELEBA', split='train', transform=transform)



    dataloader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)

    def vae_loss(recon_x, x, mu, logvar, beta=1.0, epoch=0, max_anneal_epochs=10):
        # print(recon_x.shape, x.shape)
        # print("recon_x min/max:", recon_x.min().item(), recon_x.max().item())
        # print("x min/max:", x.min().item(), x.max().item())
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # print("recon_loss:", recon_loss.item(), "kl_loss:", kl_loss.item())
        if epoch < max_anneal_epochs:
            beta = beta * epoch / max_anneal_epochs
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    losses = {
        'total': [],
        'recon': [],
        'kl': []
    }

    losses = train_vae(model, dataloader, optimizer, epochs=cfg['num_epochs'])

    torch.save(model.state_dict(), model_path)

    run.finish()

