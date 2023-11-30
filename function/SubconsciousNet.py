import torch
from torch import nn
from torch import tensor


def get_three_layer_net(input_size, hidden_size, out_size, activation=nn.ReLU):
    return nn.Sequential(nn.Linear(input_size, hidden_size), activation(), nn.Linear(hidden_size, out_size))


class SubNet(nn.Module):
    def __init__(self, input_size, output_size, deep, activation=nn.ReLU, sensitivity=0.05):
        super().__init__()
        self.register_buffer("input_num", tensor([input_size]))
        self.register_buffer("output_num", tensor([output_size]))
        self.register_buffer("s", tensor([sensitivity]))

        self.w0 = nn.Parameter(torch.randn(input_size, output_size))
        self.b0 = nn.Parameter(torch.randn(output_size))
        self.w_shape = self.w0.shape
        self.w_numel = self.w0.numel()
        self.b_shape = self.b0.shape
        self.b_numel = self.b0.numel()
        self._modules["generate_w_mask"] = get_three_layer_net(input_size, deep, self.w_numel, activation)
        self._modules["generate_b_mask"] = get_three_layer_net(input_size, deep, self.b_numel, activation)
        # print(self.w_shape)
        # print(self.w_numel)

    def forward(self, x):
        assert x.shape[0] == self.input_num or x.shape[1] == self.input_num, "wrong input!"

        if x.shape[0] != self.input_num:
            batch_size = x.shape[0]
            mask_w = torch.sigmoid(self.s * self._modules["generate_w_mask"](x).reshape(batch_size, *self.w_shape))
            mask_b = torch.sigmoid(self.s * self._modules["generate_b_mask"](x).reshape(batch_size, *self.b_shape))
            weights = mask_w * self.w0
            bias = mask_b * self.b0
            return torch.matmul(x.unsqueeze(1), weights).squeeze(1) + bias
        else:
            mask_w = torch.sigmoid(self.s * self._modules["generate_w_mask"](x).reshape(self.w_shape))
            mask_b = torch.sigmoid(self.s * self._modules["generate_b_mask"](x).reshape(self.b_shape))
            weights = mask_w * self.w0
            bias = mask_b * self.b0
            return torch.matmul(x, weights) + bias
