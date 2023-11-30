import torch
from torch import nn
from torch import tensor
from functools import reduce
from operator import mul
from Common.function.LimitedAttention import LimitedAttentionLayer


class XBlock(nn.Module):
    def __init__(self, input_shapes: tuple, output_shapes: tuple, focus_s: list, intensity=1):
        """
        Xblock为Xnet的基本单位
        :param input_shapes: 第0层的输入形状 和 前一层的输入形状
        :param output_shapes: 下一层的输出形状 和 最后一层的输出形状
        :param focus_s: 对第0层每个神经元的注视个数 和 对前一层的注视个数
        """
        assert len(focus_s) == len(input_shapes), "focus_s does not match with input_shapes!"
        assert len(input_shapes) == 2, "the length of input_shapes should be 2!"
        super().__init__()
        out_shape0_num = reduce(mul, output_shapes[0])
        out_shape1_num = reduce(mul, output_shapes[1])
        self._modules["linear"] = nn.Linear(out_shape0_num, out_shape1_num)
        for idx, (input_shape, focus) in enumerate(zip(input_shapes, focus_s)):
            self._modules[str(idx)] = LimitedAttentionLayer(input_shape, output_shapes[0], focus, intensity)

    def forward(self, x_list: list):
        """
        输入一个包括前一层输出和第0层输出的向量
        :param x_list: 第一项为第0层输入 和 第二项为前一层输入
        :return: 后一层输出 和 最后一层输出
        """
        output_next_layer = 0
        for idx, x in enumerate(x_list):
            output_next_layer += self._modules[str(idx)](x)
        output_next_layer = self.activation(output_next_layer)
        output_final_layer = self._modules["linear"](torch.flatten(output_next_layer, start_dim=1))
        return output_next_layer, output_final_layer

    def activation(self, x):  # 可自定义
        return torch.relu(x)


class XNet(nn.Module):
    def __init__(self, input_shape, direct_focus, output_shape, *modules):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._modules["direct_focus"] = LimitedAttentionLayer(input_shape, output_shape, direct_focus)
        for idx, module in enumerate(modules):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            # 注意：OrderedDict是dict的子类，是从collection中调取的
            self._modules[str(idx)] = module

    def forward(self, x):
        state = [x, 0 * x]
        y_final_layer_sum = 0
        num = 0
        for idx, block in enumerate(self._modules.values()):  # 注意：此时的_modules为一个字典
            # print(state)
            if idx == 0:  # 跳过第一个direct_focus模块
                continue
            y_next_layer, y_final_layer = block(state)
            state = [x, y_next_layer]
            y_final_layer_sum += y_final_layer
            num = idx
        y_direct = self._modules["direct_focus"](x)
        y_final_layer_avg = y_final_layer_sum / (num + 1)
        y_output = y_final_layer_avg + y_direct
        return y_output
