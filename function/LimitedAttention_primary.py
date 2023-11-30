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
    init_weights = torch.normal(0, 0.01, size=input_shape)
    init_bias = torch.normal(0, 0.01, size=input_shape)
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


def generate_neuron_layer_v2(input_shape, output_structure, focus_points):
    # 执行output_structure总数次循环,生成neurons
    total_elements = torch.prod(torch.tensor(output_structure))
    neurons = [get_neuron_params(input_shape, focus_points) for _ in range(total_elements)]
    # 将neurons按照坐标排序
    neurons.sort(key=lambda x: x[2])
    # 获得权重张量和偏置张量
    weights_list = [n[0] for n in neurons]
    bias_list = [n[1] for n in neurons]
    layer_weights = torch.stack(weights_list)
    layer_weights = layer_weights.view(*output_structure, *input_shape)
    layer_bias = torch.stack(bias_list)
    layer_bias = layer_bias.view(*output_structure, *input_shape)

    return layer_weights, layer_bias


class NeuronLayerNoBatch(nn.Module):
    def __init__(self, input_shape, output_structure, focus_points):
        super().__init__()
        self.input_shape = input_shape
        if isinstance(output_structure, int):
            self.output_structure = (output_structure,)
        else:
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


class NeuronLayer(nn.Module):
    def __init__(self, input_shape, output_structure, focus_points):
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = torch.zeros(input_shape).dim()
        if isinstance(output_structure, int):
            self.output_structure = (output_structure,)
        else:
            self.output_structure = output_structure
        self.dim_format = self.get_input_dim_format()
        self.weights, self.bias = generate_neuron_layer(input_shape, self.output_structure, focus_points)
        self.weights = nn.Parameter(self.weights.unsqueeze(-self.input_dim - 1))
        self.bias = nn.Parameter(self.bias.unsqueeze(-self.input_dim - 1))

    def forward(self, x):
        if len(x.shape) == self.input_dim:
            x = x.unsqueeze(0)
        # print(torch.sum(x * self.weights + self.bias, dim=self.dim_format).transpose(0, -1))
        return torch.sum(x * self.weights + self.bias, dim=self.dim_format).transpose(0, -1)

    def get_input_dim_format(self):
        data_dim = len(self.input_shape)
        dim_format = []
        for i in range(data_dim):
            k = -i - 1
            dim_format.append(k)
        return tuple(dim_format)
