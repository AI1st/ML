import torch
from torch import nn
from torch import tensor


# 随机生成中心点坐标
def get_centers_flatten(input_shape, focus):
    assert isinstance(input_shape, (list, tuple, torch.Tensor)) \
        , "input_shape should be a list type, tuple type or torch.Tensor type!"

    # 1.展平输入维度
    flatten_len = 1
    for i in input_shape:
        flatten_len *= i

    # 2.在展平的输入维度中随机取center
    centers = torch.randint(0, flatten_len, (focus,))
    return centers


# 张量维度与flattened张量维度相互转换(两个函数)
def flatten_index2multi_index(input_shape, flatten_index):  # flatten-index 2 multidimensional-index
    assert isinstance(input_shape, (list, tuple, torch.Tensor)) \
        , "input_shape should be a list type, tuple type or torch.Tensor type!"
    assert isinstance(flatten_index, torch.Tensor) \
        , f"flatten index should be a tensor, not {type(type(flatten_index))} !"
    assert flatten_index.dim() == 1 \
        , f"flatten_index should be a tensor, not {type(type(flatten_index))} !"

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
        , "multi_index should be a tensor!"
    assert multi_index.dim() == 2 \
        , f"flatten_index should be a tensor, not {type(type(multi_index))} !"

    # 对于shape(m,n,p)生成参考向量  (np,p,1)
    refers_base = [1]
    reversed_input_shape = list(reversed(list(input_shape)))
    for i in range(len(input_shape) - 1):
        refers_base.insert(0, reversed_input_shape[i] * refers_base[0])

    # 获得flatten_index:
    flatten_index = 0
    for i in range(len(refers_base)):
        flatten_index = flatten_index + multi_index[:, i] * refers_base[i]
    return flatten_index


class LimitedAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
