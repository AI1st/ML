import torch
from torch import nn


class Relpu(nn.Module):
    # version1
    def __init__(self):
        super().__init__()
        self.weight_a = nn.Parameter(0.01 * (torch.randn(1)))
        self.weight_b = nn.Parameter(0.01 * (torch.randn(1)) + 1)
        self.weight_c = nn.Parameter(0.01 * (torch.randn(1)) + 1)  # 暂时不用

    def forward(self, x):
        mask = x > 0
        x[mask] = self.weight_c * (x[mask] + self.weight_a * torch.pow(x[mask] + 1, self.weight_b) - self.weight_a)
        return x

    def weight_decay(self, n1, n2):
        return n1 * (self.weight_a.detach() ** 2).sum() + n2 * (self.weight_b.detach() ** 2).sum()


class Relpu2(nn.Module):
    # version2
    def __init__(self):
        super().__init__()
        self.weight_a = nn.Parameter(0.01 * (torch.randn(1)))
        self.weight_b = nn.Parameter(0.01 * (torch.randn(1)) + 1)

    def forward(self, x):
        mask = x > 0
        x[mask] = x[mask] + self.weight_a * torch.pow(x[mask] + 1, self.weight_b) - self.weight_a
        return x

    def weight_decay(self, n1, n2):
        return n1 * (self.weight_a.detach() ** 2).sum() + n2 * (self.weight_b.detach() ** 2).sum()
