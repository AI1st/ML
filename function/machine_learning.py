import numpy as np
import torch
import torch.utils.data as Data
from functools import wraps
from Common.function.graph_plot import plot
import random


def train(data_set, model, loss, optimizer, epoch_num, batch_range=(50, 50), rand=False):
    """
    函数API按照数据、模型、损失、优化(数据流动的方式)、训练次数、训练batch范围、是否在迭代过程中随机生成迭代器的方式进行
    :param data_set: 由数据特征和数据标签构成
    :param model: 输入模型
    :param loss: 损失函数
    :param optimizer: 优化器
    :param epoch_num: 迭代次数
    :param batch_range: 单batch范围
    :param rand: 是否在迭代过程中随机生成迭代器
    :return: 返回训练的历史数据
    """
    l = None
    train_ls = []
    data_iter = Data.DataLoader(data_set, batch_size=random.randint(*batch_range), shuffle=True)
    for epoch in range(epoch_num):
        if rand:
            data_iter = Data.DataLoader(data_set, batch_size=random.randint(*batch_range), shuffle=True)
        for x, y in data_iter:
            optimizer.zero_grad()
            y_hat = model(x)
            l = loss(y, y_hat)
            l.mean().backward()
            optimizer.step()
        train_ls.append(l.detach().mean())
    return train_ls


class Train:
    # 尚且存在的问题：
    # 1. 每个epoch的误差计算有误
    # 2. 显示返回的总误差有误
    # 3. 考虑关于pytorch传播模式的问题(预测模式或训练模式)
    # 4. 考虑total_loss改名为loss_in_single_batch的问题
    def __init__(self, model, optimizer, loss=None):
        """
        函数API按照模型、损失、优化(数据流动的方式)
        :param model: 输入模型
        :param loss: 损失函数
        :param optimizer: 优化器
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        # 参数初始化
        self.train_ls = []
        self.train_ls_temp = []

    def __call__(self, data_set, epoch_num, batch_range=(50, 50), rand=False):  # train
        """
        :param data_set:由数据特征和数据标签构成
        :param epoch_num:迭代次数
        :param batch_range:单batch范围
        :param rand:是否在迭代过程中随机生成迭代器
        :return: 训练历史记录
        """
        return self.train(data_set, epoch_num, batch_range, rand)

    def train(self, data_set, epoch_num, batch_range, rand=False):
        l = None  # 初始化loss
        train_ls = []
        if isinstance(batch_range, int):  # 若传入一个整数，则将其打包为元组
            batch_range = (batch_range, batch_range)

        data_iter = Data.DataLoader(data_set, batch_size=random.randint(*batch_range), shuffle=True)
        for epoch in range(epoch_num):
            if rand:
                data_iter = Data.DataLoader(data_set, batch_size=random.randint(*batch_range), shuffle=True)
            l = self.single_epoch(data_iter)
            train_ls.append(l.detach().mean())
        print(f"training loss: {train_ls[-1]}")
        self.train_ls = [*self.train_ls, *train_ls]  # 合并训练记录
        self.train_ls_temp = train_ls
        self.plot_history()
        return self.train_ls

    def single_epoch(self, data_iter):
        for k in data_iter:
            l = self.single_batch(*k)  # 此时k为列表，通过*拆包
            return l

    def single_batch(self, *k):  # 不限输入的参数
        self.optimizer.zero_grad()
        l = self.total_loss(*k)  # 此时k为元组，通过*k拆包
        l.mean().backward()
        self.optimizer.step()
        return l

    def total_loss(self, x, y):  # 默认的计算损失方式
        """
        当需要修改最终的代价函数(例如增加正则项时可以重写此方法)
        :param x: 从迭代器中拿出来数据的第一维
        :param y: 从迭代器中拿出来数据的第二维
        :return: 损失
        """
        y_hat = self.model(x)
        l = self.loss(y, y_hat)
        return l

    def plot_history(self):
        plot(np.array(range(len(self.train_ls_temp))) + len(self.train_ls) - len(self.train_ls_temp) + 1,
             [self.train_ls_temp], xlabel="epoch", ylabel="train_loss",
             title=f"training history")
        plot(np.array(range(len(self.train_ls))) + 1, [self.train_ls], xlabel="epoch", ylabel="train_loss",
             title=f"training history")

    def reset_optimizer(self, optimizer):
        self.optimizer = optimizer

    def reset_loss(self, loss):
        self.loss = loss


if __name__ == "__main__":
    # 子类集成父类后方法改写测试
    class Parent:
        def __init__(self):
            pass

        def say(self, a):
            print(a)


    class Child():
        def __init__(self):
            pass

        def say(self, a, b):
            print(a, b)


    child = Child()
    child.say(1, 2)
