import torch
from torch import nn
from d2l import torch as d2l
from Common.function.custom_layers import Relpu3


def rt_for_activation_function(model, activation_version, a_control, b_control1, b_control2):
    layers = [layer for layer in model if isinstance(layer, activation_version)]
    params1 = [layer.named_parameters() for layer in layers]
    params2 = [layer.named_parameters() for layer in layers]
    a_params = [param[1] for param_iter in params1 for param in param_iter if param[0] == "weight_a"]
    b_params = [param[1] for param_iter in params2 for param in param_iter if param[0] == "weight_b"]
    a_tensor = torch.tensor(a_params)
    b_tensor = torch.tensor(b_params)
    return a_control * torch.square(a_tensor).mean() + \
        b_control1 * torch.pow(b_control2, torch.square(b_tensor - 1)).mean()


def rt_for_activation_function_v2(model, activation_version, a_control, b_control1, b_control2, b_control3):
    layers = [layer for layer in model if isinstance(layer, activation_version)]
    params1 = [layer.named_parameters() for layer in layers]
    params2 = [layer.named_parameters() for layer in layers]
    a_params = [param[1] for param_iter in params1 for param in param_iter if param[0] == "weight_a"]
    b_params = [param[1] for param_iter in params2 for param in param_iter if param[0] == "weight_b"]
    a_tensor = torch.tensor(a_params)
    b_tensor = torch.tensor(b_params)
    return a_control * torch.square(a_tensor).mean() + \
        b_control1 * torch.pow(b_control2, b_control3 * torch.square(b_tensor - 1)).mean()


def train_relpu_like(net, train_iter, test_iter, num_epochs, lr, device,
                     a_control, b_control1, b_control2, weight_decay=0,
                     path="", model_name="", save=False):
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            # 增加了对激活函数的正则项
            l = loss(y_hat, y) + rt_for_activation_function(net, Relpu3, a_control, b_control1, b_control2)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            # 从error中扣除激活函数正则项，避免其影响最终的误差衡量
            train_l = metric[0] / metric[2] - rt_for_activation_function(net, Relpu3,
                                                                         a_control, b_control1, b_control2)
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    if save is True:
        train_acc_s = train_acc * 1000
        test_acc_s = test_acc * 1000
        torch.save(
            net.state_dict(),
            path + f"/{model_name}_tr{train_acc_s:.0f}_te{test_acc_s:.0f}"
                   f"_lr{lr}_rt{a_control}_{b_control1}_{b_control2}_wd{weight_decay}.params")


def train_relpu_like_v2(net, train_iter, test_iter, num_epochs, lr, device,
                        a_control, b_control1, b_control2, b_control3, weight_decay=0,
                        path="", model_name="", save=False):
    """

    :param net:
    :param train_iter:
    :param test_iter:
    :param num_epochs:
    :param lr:
    :param device:
    :param a_control:
    :param b_control1:
    :param b_control2:
    :param b_control3:
    :param weight_decay:
    :param path:
    :param model_name:
    :param save:
    :return:
    """
    train_l = None
    train_acc = None
    test_acc = None
    metric = None

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            # 增加了对激活函数的正则项
            l = loss(y_hat, y) + rt_for_activation_function_v2(net, Relpu3,
                                                               a_control,
                                                               b_control1, b_control2, b_control3)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()

            # 从error中扣除激活函数正则项，避免其影响最终的误差衡量
            train_l = metric[0] / metric[2] - rt_for_activation_function_v2(net, Relpu3,
                                                                            a_control,
                                                                            b_control1, b_control2, b_control3)

            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    if save is True:
        train_acc_s = train_acc * 1000
        test_acc_s = test_acc * 1000
        torch.save(
            net.state_dict(),
            path + f"/{model_name}_tr{train_acc_s:.0f}_te{test_acc_s:.0f}"
                   f"_lr{lr}_rt{a_control}_{b_control1}_{b_control2}_wd{weight_decay}.params")
