from models.VAE import VAE
from data.dataset import MNIST_Dataset, CIFAR10_Dataset, Dataset
from utils.utils import show_image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
import warnings

def vae_loss(recon_x, x, mu, logvar, beta=1.0, epoch=-1, max_anneal_epochs=10):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    if epoch != -1:
        if epoch < max_anneal_epochs:
            beta = beta * epoch / max_anneal_epochs
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def evaluate(model, dataloader) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Evaluating", leave=False):
            x = x.to(device)
            recon_x, mu, logvar, _ = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, beta=cfg['beta'])
            total_loss += loss.item() * x.size(0)
            recon_loss_total += recon_loss.item() * x.size(0)
            kl_loss_total += kl_loss.item() * x.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = recon_loss_total / len(dataloader.dataset)
    avg_kl_loss = kl_loss_total / len(dataloader.dataset)
    return avg_loss, avg_recon_loss, avg_kl_loss



def train_vae(model, train_loader, optimizer, epochs, test_loader):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
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
                "latent_space/mean": latent_mean,
                "latent_space/std": latent_std,
                "latent_space/norm_l2": latent_l2,
            })
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)

        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch/total_loss": avg_epoch_loss,
            "epoch/recon_loss": recon_loss.item(),
            "epoch/kl_loss": kl_loss.item(),
            "epoch/lr": scheduler.get_last_lr()[0]
        })
        
        if cfg['eval_every'] > 0 and (epoch + 1) % cfg['eval_every'] == 0:
            print('eval')
            val_loss, val_recon_loss, val_kl_loss = evaluate(model, test_loader)
            wandb.log({
                "val/total_loss": val_loss,
                "val/recon_loss": val_recon_loss,
                "val/kl_loss": val_kl_loss
            })
            print(f"Epoch {epoch+1}/{epochs}, : Validation - Loss: {val_loss:.4f}, recon_loss: {val_recon_loss:.4f}, kl_loss: {val_kl_loss:.4f}\n Train - Loss: {avg_epoch_loss:.4f}, recon_loss: {recon_loss.item():.4f}, kl_loss: {kl_loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_best = model.state_dict()
                torch.save(model_best, model_path)
            
        if epoch % 2 == 0:
            wandb.log({
                "input_image": wandb.Image(images[0].cpu()),
                "reconstructed_image": wandb.Image(recon[0].cpu())
            })
            
            # break
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, recon_loss: {recon_loss.item():.4f}, kl_loss: {kl_loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()
        
    return model_best


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    cfg = {
        "image_size": 64,
        "channels": 3,
        "latent_dim": 128,
        "dataset": "celeba",
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "min_lr": 1e-5,
        "n_epochs_kl_annealing": 10,
        "depth": 3,
        "beta": 2.0,
        "eval_every": 1,
        "T0_annealing": 10,
        "T_mult_annealing": 1,
        "fixed_lr_epochs": 60,
        "eta_min_lr": 1e-5
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
    test = Dataset('CELEBA', split='test', transform=transform)
    
    train_loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg['T0_annealing'], T_mult=cfg['T_mult_annealing'], eta_min=cfg['eta_min_lr']
            ),
            torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=cfg['num_epochs'] - cfg['fixed_lr_epochs']
            )
        ],
        milestones=[cfg['fixed_lr_epochs']]
    )

    model_best = train_vae(model, train_loader, optimizer, epochs=cfg['num_epochs'], test_loader=test_loader)
    
    run.finish()
