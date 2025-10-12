import torch
import torch.nn as nn
import torch.optim as optim

class simple_linear_model( nn.Module ):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(28*28, 400),
        nn.ReLU(),
        nn.Linear(400, 20)  # Output layer for mean and log-variance
    )
    def forward(self, x):
        return self.model(x)
    
    


if __name__ == "__main__":
    model = simple_linear_model()
    model.to('cuda')

    print(model)

