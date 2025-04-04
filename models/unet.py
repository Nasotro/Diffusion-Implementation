import torch
import torch.nn as nn

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

    def forward(self, x):
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            x = dropout(self.silu(norm(conv(x))))
        x_link = x.clone()
        x = self.max_pooling(x_link)
        return x, x_link

class BottleNeck(nn.Module):
    def __init__(self, input_channels:int, hidden_channels:int, conv_layers:int=2, kernel_size:int=3, dropout:float=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=1))
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden_channels) for _ in range(conv_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(conv_layers)])
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=1))
        self.up_conv = nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2)
        self.silu = nn.SiLU()

    def forward(self, x):
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            x = dropout(self.silu(norm(conv(x))))
        x = self.silu(self.up_conv(x))
        return x

class UpBlock(nn.Module):
    def __init__(self, input_channels:int, conv_layers:int=2, kernel_size:int=3, dropout:float=0.0):
        super().__init__()
        self.silu = nn.SiLU()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(input_channels * 2, input_channels, kernel_size, padding=1))
        self.norms = nn.ModuleList([nn.BatchNorm2d(input_channels) for _ in range(conv_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(conv_layers)])
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(input_channels, input_channels, kernel_size, padding=1))
        self.up_conv = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor, x_connection:torch.Tensor):
        x = torch.cat([x, x_connection], dim=1)
        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            x = dropout(self.silu(norm(conv(x))))
        x = self.silu(self.up_conv(x))
        return x

class FinalBlock(nn.Module):
    def __init__(self, input_channels, conv_layers:int=2, output_channels:int=3, dropout:float=0.0):
        super().__init__()
        self.silu = nn.SiLU()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=1))
        self.norms = nn.ModuleList([nn.BatchNorm2d(input_channels) for _ in range(conv_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(conv_layers)])
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1))
        self.final_conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor, x_connection:torch.Tensor):
        x = torch.cat([x, x_connection], dim=1)

        for conv, norm, dropout in zip(self.convs, self.norms, self.dropouts):
            x = dropout(self.silu(norm(conv(x))))

        x = self.final_conv(x)
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim, emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, t):
        return self.time_mlp(t)

class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, label):
        emb = self.emb(label)
        return self.proj(emb)


class Unet(nn.Module):
    def __init__(self, first_hidden:int = 16, depth:int = 3, time_embed_dim:int=8, label_emb_dim:int=0, num_label:int=0, initial_channels:int=3, conv_layers:int=2, dropout:float=0.0):
        super().__init__()
        self.time_emb = TimeEmbedding(time_embed_dim, time_embed_dim)
        self.label_emb = LabelEmbedding(num_label, label_emb_dim) if num_label > 0 else None
        d = depth - 1
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        self.down_blocks.append(DownBlock(initial_channels + time_embed_dim + label_emb_dim, first_hidden, conv_layers=conv_layers, dropout=dropout))
        for i in range(d):
            self.down_blocks.append(DownBlock(first_hidden * 2**i, first_hidden * 2**(i+1), conv_layers=conv_layers, dropout=dropout))

        self.bottleneck = BottleNeck(first_hidden*2**d, first_hidden*2**(d+1), conv_layers=3, dropout=dropout)

        for i in range(d, 0, -1):
            self.up_blocks.append(UpBlock(first_hidden * 2**i, conv_layers=conv_layers, dropout=dropout))

        self.final = FinalBlock(first_hidden, output_channels=initial_channels, conv_layers=conv_layers, dropout=dropout)


    def forward(self, x, time, label=None, verbose:int=0):
        time_embeddings = self.time_emb(time)
        time_embeddings = time_embeddings.unsqueeze(2).unsqueeze(3)
        time_embeddings = time_embeddings.expand(x.size(0), time_embeddings.size(1), x.size(2), x.size(3))

        if verbose==1: print(f'start with shape {x.shape}')

        if self.label_emb:
            if verbose==1: print(f'label shape : {label.shape}')
            label_embeddings = self.label_emb(label)
            if verbose==1: print(f'label_embeddings shape : {label_embeddings.shape}')
            label_embeddings = label_embeddings.unsqueeze(2).unsqueeze(2)
            if verbose==1: print(f'label_embeddings shape : {label_embeddings.shape}')
            label_embeddings = label_embeddings.expand(x.size(0), label_embeddings.size(1), x.size(2), x.size(3))
            if verbose==1: print(f'label_embeddings shape : {label_embeddings.shape}')

            x = torch.cat([x, time_embeddings, label_embeddings], dim=1)
        else:
            x = torch.cat([x, time_embeddings], dim=1)
        if verbose==1: print(f'after concatenating the timestep embedds : {x.shape}')

        skips = []
        for i, block in enumerate(self.down_blocks):
            if verbose==1: print(f'down block {i}, with shape {x.shape}')
            x, xi = block(x)
            skips.append(xi)

        x = self.bottleneck(x)
        if verbose==1: print(f'after bottleneck : shape = {x.shape}')

        for i, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if verbose==1: print(f'up block {i}, with shape {x.shape}, and skip shape : {skip.shape}')
            x = block(x, skip)

        x = self.final(x, skips[0])
        if verbose==1: print(f'after final : shape = {x.shape}')

        return x