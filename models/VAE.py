import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, input_channels:int, output_channels:int, conv_layers:int=2, kernel_size:int=3, dropout:float=0.0):
        super().__init__()
        self.silu = nn.SiLU()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(input_channels, output_channels, kernel_size, padding=1))
        self.norms = nn.ModuleList([nn.BatchNorm2d(output_channels) for _ in range(conv_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(conv_layers)])
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(output_channels, output_channels, kernel_size, padding=1))
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, verbose:int=0):
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            if verbose==1: print(f"DownBlock input shape: {x.shape}")
            x = dropout(self.silu(norm(conv(x))))
        return self.max_pooling(x)

class UpBlock(nn.Module):
    def __init__(self, latent_dim:int, conv_layers:int=2, kernel_size:int=3, dropout:float=0.0):
        super().__init__()
        self.silu = nn.SiLU()
        self.convs = nn.ModuleList()
        hidden_dim = latent_dim // 2
        self.convs.append(nn.Conv2d(latent_dim, hidden_dim, kernel_size, padding=1))
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_dim) for _ in range(conv_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(conv_layers)])
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=1))
        self.up_conv = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        
    def forward(self, x:torch.Tensor, verbose:int=0):
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            if verbose==1: print(f"UpBlock conv input shape: {x.shape}")
            x = dropout(self.silu(norm(conv(x))))
        x = self.up_conv(x)
        return x

class VAEBottleneck(nn.Module):
    def __init__(self, input_channels:int, hidden_channels:int, conv_layers:int=2, kernel_size:int=3, dropout:float=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=1))
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_channels) for _ in range(conv_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(conv_layers)])
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=1))
        self.silu = nn.SiLU()
        
    def forward(self, x):
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            x = dropout(self.silu(norm(conv(x))))
        return x
    

class VAEEncoder(nn.Module):
    def __init__(self, depth:int = 2, latent_dim:int=256, first_hidden:int = 16, initial_channels:int=3, conv_layers:int=2, dropout:float=0.0):
        super().__init__()
        self.depth = depth
        
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.down_blocks.append(DownBlock(initial_channels if i==0 else first_hidden * 2**(i-1), first_hidden * 2**i, conv_layers=conv_layers, dropout=dropout))
        
        self.bottleneck = VAEBottleneck(first_hidden * 2**(depth-1), first_hidden * 2**depth, conv_layers=conv_layers, dropout=dropout)
        
        self.fc_mu = nn.Conv2d(first_hidden * 2**depth, latent_dim, kernel_size=1)
        self.fc_logvar = nn.Conv2d(first_hidden * 2**depth, latent_dim, kernel_size=1)


    def forward(self, x:torch.Tensor, verbose:int=0):
        if verbose==1: print(f"Encoder input shape: {x.shape}")
        for i, down in enumerate(self.down_blocks):
            if verbose==1: print(f"Encoder down block {i}, input shape: {x.shape}")
            x = down(x, verbose=verbose)
            if verbose==1: print(f"Encoder down block {i}, output shape: {x.shape}")
        if verbose==1: print(f"Before bottleneck: {x.shape}")
        x = self.bottleneck(x)
        if verbose==1: print(f"After bottleneck: {x.shape}")
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        if verbose==1: print(f"Latent mu shape: {mu.shape}\nLatent logvar shape: {logvar.shape}")
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, depth:int = 2, latent_dim:int=256, initial_channels:int=3, conv_layers:int=2, dropout:float=0.0):
        super().__init__()
        self.depth = depth
        
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(UpBlock(latent_dim if i==0 else latent_dim // 2**(i), conv_layers=conv_layers, dropout=dropout))
        self.final_conv = nn.Conv2d(latent_dim // 2**(depth), initial_channels, kernel_size=1)
        
        
    def forward(self, z:torch.Tensor, verbose:int=0):
        if verbose==1: print(f"Decoder input shape: {z.shape}")
        for i, up in enumerate(self.up_blocks):
            if verbose==1: print(f"Decoder up block {i}, input shape: {z.shape}")
            z = up(z, verbose=verbose)
            if verbose==1: print(f"Decoder up block {i}, output shape: {z.shape}")
        if verbose==1: print(f"Before final conv: {z.shape}")
        z = torch.sigmoid(self.final_conv(z))
        if verbose==1: print(f"Final output shape: {z.shape}")
        return z

class VAE(nn.Module):
    def __init__(self, latent_dim:int = 256, initial_channels:int=3, depth:int=2):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(depth=depth, latent_dim=latent_dim, first_hidden=16, initial_channels=initial_channels, conv_layers=2, dropout=0.0)
        self.decoder = VAEDecoder(depth=depth, latent_dim=latent_dim, initial_channels=initial_channels, conv_layers=2, dropout=0.0)

    def encode(self, x:torch.Tensor, verbose:int=0):
        mu, logvar = self.encoder(x, verbose=verbose)
        return mu, logvar
    
    def decode(self, z:torch.Tensor, verbose:int=0):
        h = self.decoder(z, verbose=verbose)
        return h
    
    def forward(self, x:torch.Tensor, verbose:int=0):
        if verbose==1: print(f"VAE input shape: {x.shape}")
        mu, logvar = self.encode(x, verbose=verbose)
        z = self.reparametrize(mu, logvar)
        if verbose==1: print(f"Reparameterized latent shape: {z.shape}")
        reconstructed = self.decode(z, verbose=verbose)
        if verbose==1: print(f"VAE final reconstructed shape: {reconstructed.shape}")
        return reconstructed, mu, logvar, z

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
        