import os
import torch
from torch import nn


class ShowLayerOutputStatics(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        os.makedirs("layer_data_output", exist_ok=True)
        with open(f'layer{self.n}.txt', 'w') as f:
            f.write("")

    def forward(self, x):
        with open(f'layer{self.n}.txt', 'a') as f:
            f.write(str(x.detach().mean()) + "\n")
        return x