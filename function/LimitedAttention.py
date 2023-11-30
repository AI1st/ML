import torch
from torch import nn
from torch import tensor
from functools import reduce
from operator import mul


# 随机生成中心点坐标
def get_centers_flatten(input_shape, neurons):
    assert isinstance(input_shape, (list, tuple, torch.Tensor)) \
        , "input_shape should be a list type, tuple type or torch.Tensor type!"

    # 1.展平输入维度
    flatten_len = reduce(mul, input_shape)

    # 2.在展平的输入维度中随机取center
    centers = torch.randint(0, flatten_len, (neurons,))
    return centers


# 张量维度与flattened张量维度相互转换(两个函数)
def flatten_index2multi_index(input_shape, flatten_index):  # flatten-index 2 multidimensional-index
    assert isinstance(input_shape, (list, tuple, torch.Tensor)) \
        , "input_shape should be a list type, tuple type or torch.Tensor type!"
    assert isinstance(flatten_index, torch.Tensor) \
        , f"flatten index should be a tensor, not {type(type(flatten_index))} !"
    # assert flatten_index.dim() == 1 \
    #     , f"flatten_index should be a tensor, not {type(type(flatten_index))} !"

    # 对于shape(m,n,p)生成参考向量  (np,p,1)
    refers_base = [1]
    reversed_input_shape = list(reversed(list(input_shape)))
    for i in range(len(input_shape) - 1):
        refers_base.insert(0, reversed_input_shape[i] * refers_base[0])

    # 利用参考向量获得转换后的坐标
    coordinates = []
    for refer in refers_base:
        temp_coordinate = torch.floor(flatten_index / refer)
        coordinates.append(temp_coordinate)
        flatten_index = flatten_index - temp_coordinate * refer
    return torch.stack(coordinates).T


def multi_index2flatten_index(input_shape, multi_index):  # multidimensional-index 2 flatten-index
    assert isinstance(input_shape, (list, tuple, torch.Tensor)) \
        , "input_shape should be a list type, tuple type or torch.Tensor type!"
    assert isinstance(multi_index, torch.Tensor) \
        , f"multi_index should be a tensor,instead of{type(multi_index)}!"
    assert multi_index.dim() == 2 \
        , f"the dim of multi_index should be 2!"

    # 对于shape(m,n,p)生成参考向量 =>(np,p,1) [举例]
    refers_base = [1]
    reversed_input_shape = list(reversed(list(input_shape)))
    for i in range(len(input_shape) - 1):
        refers_base.insert(0, reversed_input_shape[i] * refers_base[0])

    # 获得flatten_index:
    flatten_index = 0
    for i in range(len(refers_base)):
        flatten_index = flatten_index + multi_index[:, i] * refers_base[i]
    return flatten_index


# 编写中心排序后的叠加随机张量并截断函数
def get_connections_index(input_shape, neurons, focus, intensity_index=1):
    # 获得随机连接
    centers_flatten = get_centers_flatten(input_shape, neurons)
    centers_multi = flatten_index2multi_index(input_shape, centers_flatten)
    centers_multi_sorted, _ = torch.sort(centers_multi, dim=0)
    centers_multi_sorted_repeated = centers_multi_sorted.repeat(1, focus)
    # sigma = torch.log2((torch.tensor([focus]) - 1) / 2 ** 3 + 1) * torch.sqrt(tensor([2]))
    # print("modified")
    sigma = torch.sqrt(torch.tensor([focus])) / 2
    relative_rand = intensity_index * sigma * torch.randn(centers_multi_sorted_repeated.shape)
    connections_index = centers_multi_sorted_repeated + relative_rand
    connections_index_reshape = connections_index.reshape(neurons, focus, len(input_shape))  # 这里存疑

    # 坐标截断
    min_values = torch.zeros(len(input_shape))
    max_values = torch.tensor(input_shape) - 1
    min_values = min_values.unsqueeze(0).unsqueeze(0).expand_as(connections_index_reshape)
    max_values = max_values.unsqueeze(0).unsqueeze(0).expand_as(connections_index_reshape)
    connections_index_reshape_clamped = torch.clamp(connections_index_reshape, min=min_values, max=max_values)

    # 取整
    int_connections_index_reshape_clamped = torch.round(connections_index_reshape_clamped)

    # 生成索引张量
    connections_index_multi = int_connections_index_reshape_clamped.reshape(neurons, focus * len(
        input_shape))  # 第一个维度表示神经元个数，第二个维度表示注视点个数，第三个维度表示每个注视点的坐标长度

    # 将索引张量转换为按行为单个神经元的flatten索引张量
    connections_index_multi_reshape = connections_index_multi.reshape(neurons * focus, len(input_shape))
    connections_index_flatten = multi_index2flatten_index(input_shape, connections_index_multi_reshape)
    connections_index_flatten_output = connections_index_flatten.reshape(neurons, focus)
    # print(connections_index_flatten_output)

    return connections_index_flatten_output.to(torch.int64), int_connections_index_reshape_clamped.shape  # 转换int64


def random_slice(tensor, dim):
    index = torch.randperm(tensor.size(dim))
    return torch.index_select(tensor, dim, index)


class LimitedAttentionLayerCpu(nn.Module):
    def __init__(self, input_shape, output_shape, focus, intensity=1):
        super().__init__()
        # 思路:
        # 1.创建索引张量
        # 2.创建权重张量及偏置张量
        # 3.生成保存含batch的索引字典
        # 4.定义前向传播
        # 基本输入输出尺寸初始化
        assert isinstance(input_shape, tuple), "input_shape should be tuple!"
        assert isinstance(output_shape, tuple), "output_shape should be tuple!"
        self.input_shape = input_shape
        self.output_shape = output_shape

        # 1.创建索引张量
        # 1.1计算生成的neuron个数
        neurons = reduce(mul, output_shape)
        self.neurons = neurons
        # 1.2生成随机张量
        connections_index, index_shape = get_connections_index(input_shape, neurons, focus, intensity_index=intensity)
        self.connections_index = connections_index
        self.index_shape = connections_index.shape

        # 2.生成权重张量及偏置张量
        self.weights = nn.Parameter(torch.randn(neurons, focus))
        self.bias = nn.Parameter(torch.randn(neurons))

        # 3.生成保存含batch的索引字典
        self.index_batch = {}

    def forward(self, x):
        assert x.shape[1:] == self.input_shape or x.shape[0:] == self.input_shape \
            , "the shape of input is not corresponding to the shape of index!"

        # 判断是推理模式还是batch模式
        are_equal = all(a == b for a, b in zip(x.shape, self.input_shape))
        if are_equal:
            # 推理模式：
            x = x.reshape(1, x.numel())
            batch_s = 1
        else:
            # batch模式：
            batch_s = x.shape[0]

        # 判断flatten_index_batch是否在类中的字典里存在
        if batch_s in self.index_batch:
            flatten_index_batch = self.index_batch[batch_s]
            # print("getting index from dict")  # used for test
        else:
            flatten_index = self.connections_index.reshape(1, self.connections_index.numel())
            flatten_index_batch = flatten_index.repeat(batch_s, 1).reshape(batch_s, flatten_index.numel())
            self.index_batch[batch_s] = flatten_index_batch
            # print("generating index")  # used for test

        # 前向传播
        x_flatten = torch.flatten(x, start_dim=1)
        x_selected = torch.gather(x_flatten, 1, flatten_index_batch)
        x_element_wise_product = x_selected.reshape(batch_s, *self.weights.shape) * self.weights
        x_sum_at_dim_n1 = x_element_wise_product.sum(dim=-1)
        y_without_shape = x_sum_at_dim_n1 + self.bias
        y = y_without_shape.reshape(batch_s, *self.output_shape)

        return y


class LimitedAttentionLayer(nn.Module):
    def __init__(self, input_shape, output_shape, focus, intensity=1):
        super().__init__()
        # 思路:
        # 1.创建索引张量
        # 2.创建权重张量及偏置张量
        # 3.生成保存含batch的索引字典
        # 4.定义前向传播
        # 基本输入输出尺寸初始化
        assert isinstance(input_shape, tuple), "input_shape should be tuple!"
        assert isinstance(output_shape, tuple), "output_shape should be tuple!"
        self.input_shape = input_shape
        self.output_shape = output_shape

        # 1.创建索引张量
        # 1.1计算生成的neuron个数
        neurons = reduce(mul, output_shape)
        self.register_buffer("neurons", tensor([neurons]))  # 注册为模型内部常量
        # 1.2生成随机张量
        connections_index, index_shape = get_connections_index(input_shape, neurons, focus, intensity_index=intensity)
        # 1.3注册为模型内部常量
        self.register_buffer("connections_index_numel", tensor([connections_index.numel()]))
        self.register_buffer("connections_index_standard", connections_index)
        self.register_buffer("index_shape", tensor([connections_index.shape]))

        # 2.生成权重张量及偏置张量
        self.weights = nn.Parameter(0.01 * torch.randn(neurons, focus))
        self.bias = nn.Parameter(0.01 * torch.randn(neurons))

    def forward(self, x):
        assert x.shape[1:] == self.input_shape or x.shape[0:] == self.input_shape \
            , "the shape of input is not corresponding to the shape of index!"

        # 判断是推理模式还是batch模式
        are_equal = all(a == b for a, b in zip(x.shape, self.input_shape))
        if are_equal:  # 推理模式：
            x = x.reshape(1, x.numel())
            batch_s = 1
        else:  # batch模式：
            batch_s = x.shape[0]

        # 判断flatten_index_batch是否已经生成
        if hasattr(self, f'{batch_s}'):
            flatten_index_batch = getattr(self, f"{batch_s}")  # 获取由register_buffer注册的batch_s索引
            # print("getting index from class variable")  # used for test
        else:
            flatten_index = getattr(self, "connections_index_standard").reshape(1, self.connections_index_numel)
            flatten_index_batch = flatten_index.repeat(batch_s, 1).reshape(batch_s, flatten_index.numel())
            self.register_buffer(f"{batch_s}", flatten_index_batch)  # 注册为模型内部常量
            flatten_index_batch = getattr(self, f"{batch_s}")
            # print("generating index")  # used for test

        # 前向传播
        x_flatten = torch.flatten(x, start_dim=1)
        # print(x_flatten)
        # print(flatten_index_batch)
        x_selected = torch.gather(x_flatten, 1, flatten_index_batch)
        # print(x_selected)
        x_element_wise_product = x_selected.reshape(batch_s, *self.weights.shape) * self.weights
        x_sum_at_dim_n1 = x_element_wise_product.sum(dim=-1)
        y_without_shape = x_sum_at_dim_n1 + self.bias
        y = y_without_shape.reshape(batch_s, *self.output_shape)
        # print(y)

        return y

    def load_input_and_output_size(self, input_shape, output_shape, ):  # used for loading model from dict
        self.input_shape = input_shape
        self.output_shape = output_shape


# 待完成
# 思路：通过传入的reuse_times创建复合索引层，将原索引按照neurons顺序重排，将索引由二维变为三维(reuse_times, neurons, focus)
# 此时，最终输出的
class LimitedAttentionLayerParamsReuse(nn.Module):
    def __init__(self, input_shape, output_shape, focus, reuse_times=1, intensity=1):
        super().__init__()
        # 思路:
        # 1.创建索引张量
        # 2.创建权重张量及偏置张量
        # 3.生成保存含batch的索引字典
        # 4.定义前向传播
        # 基本输入输出尺寸初始化
        assert isinstance(input_shape, tuple), "input_shape should be tuple!"
        assert isinstance(output_shape, tuple), "output_shape should be tuple!"
        self.input_shape = input_shape
        self.output_shape = output_shape

        # 1.创建索引张量
        # 1.1计算生成的neuron个数
        neurons = reduce(mul, output_shape)
        self.register_buffer("neurons", tensor([neurons]))  # 注册为模型内部常量
        self.register_buffer("reuse_times", tensor([1]) * reuse_times)  # 将reuse_times注册为模型参数
        # 1.2生成随机张量
        connections_index, index_shape = get_connections_index(input_shape, neurons, focus, intensity_index=intensity)
        # 1.3将生成的索引张量生成神经元的复用视角，并将各切片叠加为(reuse_times, neurons, focus)的形式
        # 生成reuse_times-1片切片并合成大张量
        index_shuffle_list = [connections_index]
        for i in range(reuse_times - 1):
            index_shuffle_list.append(random_slice(connections_index, 0))
        connections_index_shuffle = torch.stack(index_shuffle_list)
        # 1.4注册为模型内部常量
        self.register_buffer("connections_index_numel", tensor([connections_index_shuffle.numel()]))
        self.register_buffer("connections_index_standard", connections_index_shuffle)
        self.register_buffer("index_shape", tensor([connections_index_shuffle.shape]))

        # 2.生成权重张量及偏置张量
        self.weights = nn.Parameter(0.01 * torch.randn(neurons, focus))
        self.bias = nn.Parameter(0.01 * torch.randn(neurons))

    def forward(self, x):
        assert x.shape[1:] == self.input_shape or x.shape[0:] == self.input_shape \
            , "the shape of input is not corresponding to the shape of index!"

        # 判断是推理模式还是batch模式
        are_equal = all(a == b for a, b in zip(x.shape, self.input_shape))
        if are_equal:  # 推理模式：
            x = x.reshape(1, x.numel())
            batch_s = 1
        else:  # batch模式：
            batch_s = x.shape[0]

        # 判断flatten_index_batch是否已经生成
        if hasattr(self, f'{batch_s}'):
            flatten_index_batch = getattr(self, f"{batch_s}")  # 获取由register_buffer注册的batch_s索引
            # print("getting index from class variable")  # used for test
        else:
            flatten_index = getattr(self, "connections_index_standard").reshape(1, self.connections_index_numel)
            flatten_index_batch = flatten_index.repeat(batch_s, 1).reshape(batch_s, flatten_index.numel())
            self.register_buffer(f"{batch_s}", flatten_index_batch)  # 注册为模型内部常量
            flatten_index_batch = getattr(self, f"{batch_s}")
            # print("generating index")  # used for test

        # 前向传播
        x_flatten = torch.flatten(x, start_dim=1)
        # print(x_flatten)
        # print(flatten_index_batch)
        x_selected = torch.gather(x_flatten, 1, flatten_index_batch)
        # print(x_selected)
        x_element_wise_product = x_selected.reshape(batch_s, self.reuse_times, *self.weights.shape) * self.weights
        x_sum_at_dim_n1 = x_element_wise_product.sum(dim=-1)
        y_without_shape = x_sum_at_dim_n1 + self.bias
        y = y_without_shape.reshape(batch_s, self.reuse_times, *self.output_shape)
        # print(y)

        return y

    def load_input_and_output_size(self, input_shape, output_shape, ):  # used for loading model from dict
        self.input_shape = input_shape
        self.output_shape = output_shape
