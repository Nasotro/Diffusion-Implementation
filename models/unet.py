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

class Unet(nn.Module):
    def __init__(self, first_hidden:int = 16, depth:int = 3):
        super().__init__()
        d = depth - 1
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        self.down_blocks.append(DownBlock(3, first_hidden))
        for i in range(d):
            self.down_blocks.append(DownBlock(first_hidden * 2**i, first_hidden * 2**(i+1)))
        
        self.bottleneck = BottleNeck(first_hidden*2**d, first_hidden*2**(d+1))
        
        for i in range(d, 0, -1):
            self.up_blocks.append(UpBlock(first_hidden * 2**i))
            
        self.final = FinalBlock(first_hidden, 3)
        

    def forward(self, x):
        links = []
        for i, block in enumerate(self.down_blocks):
            print(f'down block {i}, with shape {x.shape}')
            x, xi = block(x)
            links.append(xi)
        
        x = self.bottleneck(x)
        print(f'after bottleneck : shape = {x.shape}')
        
        for i, block in enumerate(self.up_blocks):
            print(f'up block {i}, with shape {x.shape}')
            x = block(x, links.pop())
        
        x = self.final(x, links[0])
        print(f'after final : shape = {x.shape}')
        
        return x

