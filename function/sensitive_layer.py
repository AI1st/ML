import torch
from torch import nn


class Sensitivity(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.sensitivity = s

    def forward(self, x):
        return self.sensitivity * x
