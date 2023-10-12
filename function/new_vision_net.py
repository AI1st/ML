import torch
from torch import nn
from torch import tensor


def generate_center(input_shape):
    center = []
    for index in input_shape:
        coordinate = torch.randint(0, index, size=(1,))
        center.append(coordinate)
    return center


def generate_params_mask(zero_mask, center, m, std):
    d = zero_mask.dim()
    if isinstance(std, torch.Tensor):
        std = std.item()
    # print(d)
    size = (m, d)
    # print(size)
    # 生成符合正态分布的随机数
    random_coords = torch.normal(0, std=std, size=size, device=zero_mask.device)
    # 将随机数添加到中心点上
    centered_coords = random_coords + torch.tensor(center, device=zero_mask.device)
    # 四舍五入并转换为整数
    indices = torch.round(centered_coords).long()
    # 将超出范围的索引截断
    for i in range(d):
        indices[:, i] = indices[:, i].clamp(min=0, max=zero_mask.size(i) - 1)
    # 将全零张量在这些坐标位置的值设置为1
    zero_mask[tuple(indices.t().tolist())] = 1
    zero_mask[tuple(center)] = 1
    return zero_mask


def get_neuron_params(input_shape, connected_nums):
    # 1.生成全零张量：
    zeros_filter = torch.zeros(input_shape)
    # 2.生成初始权重矩阵和偏置
    init_weights = torch.randn(input_shape)
    init_bias = torch.randn(input_shape)
    # 3.选择中心：
    center = generate_center(input_shape)
    # print(center)
    std = torch.sqrt(tensor([connected_nums])) / 2
    # 4.获得全零filter并得到筛选后的weights和bias
    zeros_filter = generate_params_mask(zeros_filter, center, connected_nums, std)
    weights = init_weights * zeros_filter  # 获得选择性的矩阵
    bias = init_bias * zeros_filter  # 获得选择性的偏置
    return weights, bias, center


def generate_neuron_layer(input_shape, output_structure, focus_points):
    # 生成两个四维张量
    weights_layer = torch.zeros(*output_structure, *input_shape)
    bias_layer = torch.zeros(*output_structure, *input_shape)
    # 获得neuron
    neurons = []
    # 执行output_structure总数次循环,生成neurons
    total_elements = torch.prod(torch.tensor(output_structure))
    # print(total_elements)
    for n in range(total_elements):
        # 获取权重张量和中心坐标
        weights, bias, center = get_neuron_params(input_shape, focus_points)
        # 保存权重张量和中心坐标
        neurons.append((weights, bias, center))
    neurons.sort(key=lambda x: x[2])
    weights_list = []
    bias_list = []
    for n in neurons:
        weights_list.append(n[0])
        bias_list.append(n[1])
    layer_weights = torch.stack(weights_list)
    layer_weights = layer_weights.view(*output_structure, *input_shape)
    layer_bias = torch.stack(bias_list)
    layer_bias = layer_bias.view(*output_structure, *input_shape)

    return layer_weights, layer_bias


class NeuronLayer(nn.Module):
    def __init__(self, input_shape, output_structure, focus_points):
        super().__init__()
        self.input_shape = input_shape
        self.output_structure = output_structure
        self.dim_format = self.get_input_dim_format()
        self.weights, self.bias = generate_neuron_layer(input_shape, output_structure, focus_points)
        self.weights = nn.Parameter(self.weights)
        self.bias = nn.Parameter(self.bias)

    def forward(self, x):
        return torch.sum(x * self.weights + self.bias, dim=self.dim_format)

    def get_input_dim_format(self):
        data_dim = len(self.input_shape)
        dim_format = []
        for i in range(data_dim):
            k = -i - 1
            dim_format.append(k)
        return tuple(dim_format)
