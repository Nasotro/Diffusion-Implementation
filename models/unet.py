import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, input_channels:int, output_channels:int, conv_layers:int=2, kernel_size:int=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(input_channels, output_channels, kernel_size, padding=1))
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(output_channels, output_channels, kernel_size, padding=1))
        self.max_pooling = nn.MaxPool2d(kernel_size=1, stride=2)

    def forward(self, x):
        for conv in self.convs:
            x = self.relu(conv(x))
        x_link = x.clone()
        x = self.max_pooling(x_link)
        return x, x_link

class BottleNeck(nn.Module):
    def __init__(self, input_channels:int, hidden_channels:int, conv_layers:int=2, kernel_size:int=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=1))
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=1))
        self.up_conv = nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for conv in self.convs:
            x = self.relu(conv(x))
        x = self.relu(self.up_conv(x))
        return x

class UpBlock(nn.Module):
    def __init__(self, input_channels:int, conv_layers:int=2, kernel_size:int=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(input_channels * 2, input_channels, kernel_size, padding=1))
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(input_channels, input_channels, kernel_size, padding=1))
        self.up_conv = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)

    def forward(self, x:torch.Tensor, x_connection:torch.Tensor):
        x = torch.cat([x, x_connection], dim=1)
        for conv in self.convs:
            x = self.relu(conv(x))
        x = self.relu(self.up_conv(x))
        return x

class FinalBlock(nn.Module):
    def __init__(self, input_channels, conv_layers:int=2, output_channels:int=3):
        super().__init__()
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=1))  
        for _ in range(conv_layers - 1):
            self.convs.append(nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1))
        self.final_conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor, x_connection:torch.Tensor):
        x = torch.cat([x, x_connection], dim=1)
        
        for conv in self.convs:
            x = self.relu(conv(x))
        
        x = self.relu(self.final_conv(x))
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

class Unet(nn.Module):
    def __init__(self, first_hidden:int = 16, depth:int = 3, time_embed_dim:int=8):
        super().__init__()
        self.time_emb = SinusoidalPositionEmbeddings(time_embed_dim)
        d = depth - 1
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        self.down_blocks.append(DownBlock(3 + time_embed_dim, first_hidden))
        for i in range(d):
            self.down_blocks.append(DownBlock(first_hidden * 2**i, first_hidden * 2**(i+1)))
        
        self.bottleneck = BottleNeck(first_hidden*2**d, first_hidden*2**(d+1))
        
        for i in range(d, 0, -1):
            self.up_blocks.append(UpBlock(first_hidden * 2**i))
            
        self.final = FinalBlock(first_hidden, 3)
        

    def forward(self, x, time):
        time_embeddings = self.time_emb(time)
        time_embeddings = time_embeddings.unsqueeze(2).unsqueeze(3)
        time_embeddings = time_embeddings.expand(x.size(0), time_embeddings.size(1), x.size(2), x.size(3))
    
        print(f'start with shape {x.shape}')
        
        x = torch.cat([x, time_embeddings], dim=1)
        print(f'after concatenating the timestep embedds : {x.shape}')
        
        skips = []
        for i, block in enumerate(self.down_blocks):
            print(f'down block {i}, with shape {x.shape}')
            x, xi = block(x)
            skips.append(xi)
        
        x = self.bottleneck(x)
        print(f'after bottleneck : shape = {x.shape}')
        
        for i, block in enumerate(self.up_blocks):
            print(f'up block {i}, with shape {x.shape}')
            x = block(x, skips.pop())
        
        x = self.final(x, skips[0])
        print(f'after final : shape = {x.shape}')
        
        return x

